from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config.settings import AppSettings, settings
from src.agent.agent import MetrologyAgentRunner
from src.models.schemas import EvaluationCase, EvaluationCaseResult, EvaluationSummary
from src.rules.validator import canonicalize_instrument_type, normalize_unit


def load_evaluation_cases(path: Path | None = None) -> list[EvaluationCase]:
    cases_path = path or settings.evaluation_cases_path
    if not cases_path.exists():
        raise FileNotFoundError(f"测试问题 JSON 不存在: {cases_path}")
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(item) for item in data]


def _almost_equal(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return str(left) == str(right)


def _parameters_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key == "instrument_type":
            if canonicalize_instrument_type(actual_value) != canonicalize_instrument_type(expected_value):
                return False
        elif key == "unit":
            if normalize_unit(actual_value) != normalize_unit(expected_value):
                return False
        elif not _almost_equal(actual_value, expected_value):
            return False
    return True


def _tool_outputs_ok(trace: list[Any], expected_tools: list[str]) -> bool:
    for item in trace:
        if item.tool_name in expected_tools and isinstance(item.tool_output, dict) and item.tool_output.get("error"):
            return False
    return True


def _passed_recommendation_exists(result: Any) -> bool:
    return any(rec.validation.passed for rec in result.recommendations)


def run_evaluation(
    app_settings: AppSettings = settings,
    cases: list[EvaluationCase] | None = None,
) -> EvaluationSummary:
    evaluation_cases = cases or load_evaluation_cases(app_settings.evaluation_cases_path)
    runner = MetrologyAgentRunner(app_settings)
    case_results: list[EvaluationCaseResult] = []

    for case in evaluation_cases:
        started = time.perf_counter()
        error = None
        try:
            result = runner.run(case.question)
        except Exception as exc:
            result = None
            error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000

        if result is None:
            case_results.append(
                EvaluationCaseResult(
                    id=case.id,
                    question=case.question,
                    expected_tools=case.expected_tools,
                    actual_tools=[],
                    tool_call_success=0,
                    end_to_end_success=0,
                    response_time_ms=round(elapsed_ms, 2),
                    error=error,
                )
            )
            continue

        actual_tools = [item.tool_name for item in result.trace]
        tools_present = all(tool_name in actual_tools for tool_name in case.expected_tools)
        params_ok = (
            result.parameters is not None
            and _parameters_match(result.parameters.model_dump(), case.expected_parameters)
        )
        tool_call_success = int(tools_present and _tool_outputs_ok(result.trace, case.expected_tools) and params_ok)

        recommended_ids = {rec.instrument.id for rec in result.recommendations}
        expected_device_hit = bool(case.expected_device_ids) and set(case.expected_device_ids).issubset(
            recommended_ids
        )
        recommendation_ok = (
            expected_device_hit if case.expected_device_ids else True
        ) or _passed_recommendation_exists(result)
        citation_ok = bool(result.citations)
        end_to_end_success = int(
            tool_call_success == 1
            and recommendation_ok
            and citation_ok
            and not result.error
        )
        failure_reasons = []
        if not tools_present:
            missing = [tool_name for tool_name in case.expected_tools if tool_name not in actual_tools]
            failure_reasons.append(f"missing_tools={missing}")
        if not _tool_outputs_ok(result.trace, case.expected_tools):
            failure_reasons.append("tool_output_error")
        if not params_ok:
            failure_reasons.append("parameters_mismatch")
        if not recommendation_ok:
            failure_reasons.append("no_valid_recommendation")
        if not citation_ok:
            failure_reasons.append("no_citation")
        if result.error:
            failure_reasons.append(f"agent_error={result.error}")

        case_results.append(
            EvaluationCaseResult(
                id=case.id,
                question=case.question,
                expected_tools=case.expected_tools,
                actual_tools=actual_tools,
                tool_call_success=tool_call_success,
                end_to_end_success=end_to_end_success,
                response_time_ms=round(elapsed_ms, 2),
                error=result.error,
                recommendation_success=int(recommendation_ok),
                citation_success=int(citation_ok),
                expected_device_hit=int(expected_device_hit),
                failure_reason="; ".join(failure_reasons),
            )
        )

    total = len(case_results)
    tool_successes = sum(item.tool_call_success for item in case_results)
    e2e_successes = sum(item.end_to_end_success for item in case_results)
    total_tool_calls = sum(len(item.actual_tools) for item in case_results)
    total_response_time = sum(item.response_time_ms for item in case_results)

    return EvaluationSummary(
        total_cases=total,
        tool_call_success_rate=tool_successes / total if total else 0.0,
        end_to_end_success_rate=e2e_successes / total if total else 0.0,
        average_tool_calls=total_tool_calls / total if total else 0.0,
        average_response_time_ms=total_response_time / total if total else 0.0,
        case_results=case_results,
    )


if __name__ == "__main__":
    summary = run_evaluation()
    print(summary.model_dump_json(indent=2))
