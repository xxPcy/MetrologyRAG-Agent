from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from config.settings import AppSettings, settings
from src.agent.prompts import SYSTEM_PROMPT
from src.models.schemas import (
    AgentRunResult,
    ConversationState,
    InstrumentParameters,
    Recommendation,
    SearchResult,
    ToolTrace,
)
from src.tools.toolkit import build_tools
from src.utils.logger import get_logger, shorten_for_log


logger = get_logger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _as_json_text(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, default=str)


class MetrologyAgentRunner:
    """LangChain tool-calling runner with an API-less deterministic fallback."""

    def __init__(self, app_settings: AppSettings = settings) -> None:
        self.settings = app_settings
        self.tools = build_tools(app_settings)
        self.tool_map: dict[str, BaseTool] = {tool.name: tool for tool in self.tools}

    def run(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        structured_state: ConversationState | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        logger.info("Agent request: %s", shorten_for_log(question, 300))
        state = self._coerce_state(structured_state)
        if not self.settings.llm_configured:
            return self.run_deterministic_flow(
                question,
                note="未配置 LLM API，已使用本地规则化演示流程执行同一组工具。",
                structured_state=state,
            )

        try:
            return self.run_llm_tool_calling(question, chat_history=chat_history, structured_state=state)
        except Exception as exc:
            logger.exception("LLM agent failed: %s", exc)
            return self.run_deterministic_flow(
                question,
                note="LLM 请求失败，已切换到本地规则化演示流程。",
                structured_state=state,
            )

    def run_llm_tool_calling(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        structured_state: ConversationState | None = None,
    ) -> AgentRunResult:
        from langchain_openai import ChatOpenAI

        llm_kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "api_key": self.settings.llm_api_key,
            "temperature": self.settings.llm_temperature,
        }
        if self.settings.llm_base_url:
            llm_kwargs["base_url"] = self.settings.llm_base_url

        llm = ChatOpenAI(**llm_kwargs)
        llm_with_tools = llm.bind_tools(self.tools)
        messages = self._build_messages(question, chat_history, structured_state)
        trace: list[ToolTrace] = []

        for _ in range(self.settings.agent_max_steps):
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                self._complete_missing_core_flow(question, trace, structured_state)
                return self._result_from_trace(
                    question,
                    str(response.content),
                    trace,
                    structured_state,
                )

            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args") or {}
                tool_id = call.get("id") or f"call_{len(trace) + 1}"
                output = self._invoke_tool(tool_name, tool_args, trace)
                if tool_name == "query_instrument_catalog" and isinstance(output, list):
                    parameters = tool_args.get("parameters")
                    if isinstance(parameters, dict):
                        for candidate in output[:10]:
                            self._invoke_tool(
                                "validate_instrument",
                                {"parameters": parameters, "instrument": candidate},
                                trace,
                            )
                messages.append(
                    ToolMessage(
                        content=_as_json_text(output),
                        tool_call_id=tool_id,
                    )
                )

        answer = "Agent 达到最大工具调用步数，已停止执行。请缩小问题范围或提高 AGENT_MAX_STEPS。"
        self._complete_missing_core_flow(question, trace, structured_state)
        return self._result_from_trace(question, answer, trace, structured_state)

    def _complete_missing_core_flow(
        self,
        question: str,
        trace: list[ToolTrace],
        structured_state: ConversationState | None = None,
    ) -> None:
        """Programmatic guardrail for cases where the LLM skips required tools."""
        if not any(item.tool_name == "search_metrology_standard" for item in trace):
            self._invoke_tool(
                "search_metrology_standard",
                {"query": question, "top_k": self.settings.retrieval_top_k},
                trace,
            )

        citations_raw = []
        for item in trace:
            if item.tool_name == "search_metrology_standard" and isinstance(item.tool_output, list):
                citations_raw = item.tool_output
                break
        context = "\n\n".join(str(item.get("content", "")) for item in citations_raw[:3])

        params_raw = None
        for item in reversed(trace):
            if item.tool_name == "extract_parameters" and isinstance(item.tool_output, dict):
                params_raw = item.tool_output
                break
        if params_raw is None:
            params_raw = self._invoke_tool(
                "extract_parameters",
                {"text": question, "context": context},
                trace,
            )

        if not isinstance(params_raw, dict):
            return
        parameters = InstrumentParameters.model_validate(params_raw)
        parameters = self._merge_parameters_with_state(question, parameters, structured_state)
        if parameters.model_dump() != params_raw:
            self._append_trace(
                trace,
                "merge_conversation_state",
                {
                    "question": question,
                    "current_parameters": params_raw,
                    "previous_parameters": (
                        structured_state.last_parameters.model_dump()
                        if structured_state and structured_state.last_parameters
                        else None
                    ),
                },
                parameters.model_dump(),
            )
        if not parameters.has_required_selection_fields():
            return

        candidates = None
        for item in reversed(trace):
            if item.tool_name == "query_instrument_catalog" and isinstance(item.tool_output, list):
                candidates = item.tool_output
                break
        if candidates is None:
            candidates = self._invoke_tool(
                "query_instrument_catalog",
                {"parameters": parameters.model_dump(), "limit": 10},
                trace,
            )
        if not isinstance(candidates, list):
            return

        validated_ids = {
            item.tool_input.get("instrument", {}).get("id")
            for item in trace
            if item.tool_name == "validate_instrument" and isinstance(item.tool_input, dict)
        }
        for candidate in candidates[:10]:
            if not isinstance(candidate, dict) or candidate.get("id") in validated_ids:
                continue
            self._invoke_tool(
                "validate_instrument",
                {"parameters": parameters.model_dump(), "instrument": candidate},
                trace,
            )

        if not any(item.tool_name == "recommend_instruments" for item in trace):
            self._invoke_tool(
                "recommend_instruments",
                {
                    "parameters": parameters.model_dump(),
                    "candidates": candidates,
                    "limit": 5,
                },
                trace,
            )

    def run_deterministic_flow(
        self,
        question: str,
        note: str | None = None,
        structured_state: ConversationState | dict[str, Any] | None = None,
    ) -> AgentRunResult:
        state = self._coerce_state(structured_state)
        trace: list[ToolTrace] = []
        citations_raw = self._invoke_tool(
            "search_metrology_standard",
            {"query": self._contextualized_query(question, state), "top_k": self.settings.retrieval_top_k},
            trace,
        )
        citations = [SearchResult.model_validate(item) for item in citations_raw or []]

        context = "\n\n".join(item.content for item in citations[:3])
        params_raw = self._invoke_tool(
            "extract_parameters",
            {"text": question, "context": context},
            trace,
        )
        parameters = self._merge_parameters_with_state(
            question,
            InstrumentParameters.model_validate(params_raw),
            state,
        )
        if parameters.model_dump() != params_raw:
            self._append_trace(
                trace,
                "merge_conversation_state",
                {
                    "question": question,
                    "current_parameters": params_raw,
                    "previous_parameters": state.last_parameters.model_dump() if state.last_parameters else None,
                },
                parameters.model_dump(),
            )

        recommendations: list[Recommendation] = []
        if parameters.has_required_selection_fields():
            candidates = self._invoke_tool(
                "query_instrument_catalog",
                {"parameters": parameters.model_dump(), "limit": 10},
                trace,
            )
            for candidate in candidates[:10]:
                self._invoke_tool(
                    "validate_instrument",
                    {"parameters": parameters.model_dump(), "instrument": candidate},
                    trace,
                )
            recs_raw = self._invoke_tool(
                "recommend_instruments",
                {
                    "parameters": parameters.model_dump(),
                    "candidates": candidates,
                    "limit": 5,
                },
                trace,
            )
            recommendations = [Recommendation.model_validate(item) for item in recs_raw]

        answer = self._compose_answer(question, parameters, recommendations, citations, note)
        return AgentRunResult(
            question=question,
            answer=answer,
            trace=trace,
            parameters=parameters,
            recommendations=recommendations,
            citations=citations,
            structured_state=self._updated_state(state, parameters, recommendations, citations),
        )

    def _coerce_state(self, structured_state: ConversationState | dict[str, Any] | None) -> ConversationState:
        if isinstance(structured_state, ConversationState):
            return structured_state
        if isinstance(structured_state, dict):
            return ConversationState.model_validate(structured_state)
        return ConversationState()

    def _state_summary(self, structured_state: ConversationState | None) -> str:
        if not structured_state or not structured_state.last_parameters:
            return ""
        params = structured_state.last_parameters
        parts = []
        if params.instrument_type:
            parts.append(f"仪器类型={params.instrument_type}")
        if params.range_min is not None and params.range_max is not None and params.unit:
            parts.append(f"量程={params.range_min}～{params.range_max} {params.unit}")
        if params.accuracy_class is not None:
            parts.append(f"准确度={params.accuracy_class} 级")
        return "；".join(parts)

    def _contextualized_query(self, question: str, structured_state: ConversationState | None) -> str:
        summary = self._state_summary(structured_state)
        if not summary:
            return question
        return f"{question}\n上一轮已确认参数：{summary}"

    def _build_messages(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None,
        structured_state: ConversationState | None,
    ) -> list[Any]:
        system_prompt = SYSTEM_PROMPT
        summary = self._state_summary(structured_state)
        if summary:
            system_prompt += (
                "\n\n当前多轮对话的结构化上下文："
                f"{summary}。"
                "如果用户追问中省略了仪器类型、量程或准确度，可优先继承该上下文；"
                "如果用户给出新参数，则以用户本轮输入为准。"
            )
        messages: list[Any] = [SystemMessage(content=system_prompt)]
        for item in (chat_history or [])[-8:]:
            role = item.get("role")
            content = item.get("content", "")
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=question))
        return messages

    def _should_inherit_state(
        self,
        question: str,
        parameters: InstrumentParameters,
        structured_state: ConversationState | None,
    ) -> bool:
        if not structured_state or not structured_state.last_parameters:
            return False
        if parameters.has_required_selection_fields():
            return False
        follow_up_cues = [
            "那",
            "如果",
            "换成",
            "改成",
            "同样",
            "继续",
            "这个",
            "刚才",
            "上一",
            "它",
            "呢",
        ]
        has_partial_numeric = any(
            value is not None
            for value in [parameters.range_min, parameters.range_max, parameters.accuracy_class]
        )
        has_follow_up_cue = any(cue in question for cue in follow_up_cues)
        return has_partial_numeric or has_follow_up_cue

    def _merge_parameters_with_state(
        self,
        question: str,
        parameters: InstrumentParameters,
        structured_state: ConversationState | None,
    ) -> InstrumentParameters:
        if not self._should_inherit_state(question, parameters, structured_state):
            return parameters
        previous = structured_state.last_parameters if structured_state else None
        if previous is None:
            return parameters

        current_type = parameters.instrument_type
        previous_type = previous.instrument_type
        type_changed = bool(current_type and previous_type and current_type != previous_type)
        return InstrumentParameters(
            instrument_type=current_type or previous.instrument_type,
            range_min=parameters.range_min if parameters.range_min is not None or type_changed else previous.range_min,
            range_max=parameters.range_max if parameters.range_max is not None or type_changed else previous.range_max,
            unit=parameters.unit or (None if type_changed else previous.unit),
            accuracy_class=(
                parameters.accuracy_class
                if parameters.accuracy_class is not None or type_changed
                else previous.accuracy_class
            ),
            accuracy_text=parameters.accuracy_text or (None if type_changed else previous.accuracy_text),
            raw_text=parameters.raw_text or question,
        )

    def _updated_state(
        self,
        previous_state: ConversationState,
        parameters: InstrumentParameters | None,
        recommendations: list[Recommendation],
        citations: list[SearchResult],
    ) -> ConversationState:
        return ConversationState(
            last_parameters=parameters or previous_state.last_parameters,
            last_recommendations=recommendations or previous_state.last_recommendations,
            last_citations=citations or previous_state.last_citations,
            user_constraints=previous_state.user_constraints,
        )

    def _append_trace(
        self,
        trace: list[ToolTrace],
        tool_name: str,
        tool_input: dict[str, Any],
        tool_output: Any,
    ) -> None:
        trace.append(
            ToolTrace(
                step=len(trace) + 1,
                tool_name=tool_name,
                tool_input=_jsonable(tool_input),
                tool_output=_jsonable(tool_output),
                elapsed_ms=0.0,
            )
        )

    def _invoke_tool(
        self,
        tool_name: str | None,
        tool_args: dict[str, Any],
        trace: list[ToolTrace],
    ) -> Any:
        if not tool_name or tool_name not in self.tool_map:
            output = {"error": f"未知工具: {tool_name}"}
            trace.append(
                ToolTrace(
                    step=len(trace) + 1,
                    tool_name=tool_name or "unknown",
                    tool_input=tool_args,
                    tool_output=output,
                    elapsed_ms=0.0,
                )
            )
            return output

        start = time.perf_counter()
        try:
            output = self.tool_map[tool_name].invoke(tool_args)
        except Exception as exc:
            logger.exception("Tool %s failed: %s", tool_name, exc)
            output = {"error": str(exc)}
        elapsed_ms = (time.perf_counter() - start) * 1000
        trace.append(
            ToolTrace(
                step=len(trace) + 1,
                tool_name=tool_name,
                tool_input=_jsonable(tool_args),
                tool_output=_jsonable(output),
                elapsed_ms=round(elapsed_ms, 2),
            )
        )
        logger.info(
            "Tool call | %s | input=%s | output=%s | %.2fms",
            tool_name,
            shorten_for_log(tool_args, 500),
            shorten_for_log(output, 800),
            elapsed_ms,
        )
        return output

    def _result_from_trace(
        self,
        question: str,
        answer: str,
        trace: list[ToolTrace],
        structured_state: ConversationState | None = None,
    ) -> AgentRunResult:
        state = self._coerce_state(structured_state)
        parameters = None
        recommendations: list[Recommendation] = []
        citations: list[SearchResult] = []
        for item in trace:
            if item.tool_name in {"extract_parameters", "merge_conversation_state"} and isinstance(
                item.tool_output, dict
            ):
                parameters = InstrumentParameters.model_validate(item.tool_output)
            elif item.tool_name == "recommend_instruments" and isinstance(item.tool_output, list):
                recommendations = [Recommendation.model_validate(rec) for rec in item.tool_output]
            elif item.tool_name == "search_metrology_standard" and isinstance(item.tool_output, list):
                citations = [SearchResult.model_validate(result) for result in item.tool_output]

        return AgentRunResult(
            question=question,
            answer=answer,
            trace=trace,
            parameters=parameters,
            recommendations=recommendations,
            citations=citations,
            structured_state=self._updated_state(state, parameters, recommendations, citations),
        )

    def _compose_answer(
        self,
        question: str,
        parameters: InstrumentParameters,
        recommendations: list[Recommendation],
        citations: list[SearchResult],
        note: str | None,
    ) -> str:
        lines: list[str] = []
        if note:
            lines.append(note)

        if parameters.has_required_selection_fields():
            lines.append(
                "已识别被检仪器参数："
                f"{parameters.instrument_type}，{parameters.range_min}～{parameters.range_max} {parameters.unit}，"
                f"{parameters.accuracy_class} 级。"
            )
            if recommendations:
                lines.append("推荐优先选择以下标准器：")
                for index, rec in enumerate(recommendations, start=1):
                    inst = rec.instrument
                    lines.append(
                        f"{index}. {inst.id} | {inst.name} | {inst.model} | "
                        f"{inst.range_min}～{inst.range_max} {inst.unit} | {inst.accuracy_class} 级。"
                    )
                lines.append("以上推荐已通过 Python 规则引擎校验；规则来自 config/rules.yaml 的 Demo 配置。")
            else:
                lines.append("未找到通过当前 Demo 规则校验的标准器，请补充或调整标准器目录。")
        else:
            missing = []
            if not parameters.instrument_type:
                missing.append("被检仪器类型")
            if parameters.range_min is None or parameters.range_max is None or not parameters.unit:
                missing.append("量程和单位")
            if parameters.accuracy_class is None:
                missing.append("准确度等级")
            if missing:
                lines.append("当前问题不足以完成标准器选型，缺少：" + "、".join(missing) + "。")

        if citations:
            lines.append("标准文档引用：")
            for item in citations[:3]:
                page_text = f"第 {item.page} 页" if item.page else "页码未知"
                section = f"，章节：{item.section}" if item.section else ""
                lines.append(f"- {item.source}，{page_text}{section}")
        else:
            lines.append("未检索到可引用的标准文档片段。")

        if "标准器" not in question and "检定" not in question and "校准" not in question:
            lines.append("可继续追问具体量程、准确度等级和仪器类型，我会补齐选型流程。")
        return "\n".join(lines)
