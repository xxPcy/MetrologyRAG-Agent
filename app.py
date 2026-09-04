from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
import streamlit as st

from config.settings import get_settings
from src.agent.agent import MetrologyAgentRunner
from src.evaluation.evaluator import run_evaluation
from src.ingestion.pdf_to_markdown import convert_pdf_directory
from src.ingestion.vector_store import build_vector_store, count_markdown_chunks, get_chroma_count
from src.models.schemas import AgentRunResult
from src.utils.logger import get_logger


logger = get_logger(__name__)
settings = get_settings()
PDF_PREVIEW_ZOOM = 1.5
PDF_PREVIEW_HEIGHT = 760


st.set_page_config(
    page_title="MetrologyRAG-Agent",
    layout="wide",
)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def inject_styles() -> None:
    style_path = Path(__file__).parent / "assets" / "styles.css"
    if style_path.exists():
        st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _short_label(value: str, fallback: str) -> str:
    text = value.strip() if value else fallback
    return text if len(text) <= 48 else f"{text[:45]}..."


def render_header() -> None:
    llm_label = _short_label(settings.llm_model if settings.llm_configured else "", "本地规则流程")
    embedding_label = _short_label(
        settings.local_embedding_model or settings.embedding_model or "",
        "Hash fallback",
    )
    st.markdown(
        f"""
        <div class="app-header">
          <div class="app-title">
            <h1>计量标准器选型工作台</h1>
            <p>Metrological Standard Selection Platform</p>
          </div>
          <div class="runtime-pills">
            <span>LLM · {escape(llm_label)}</span>
            <span>Embedding · {escape(embedding_label)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_catalog(path: str) -> pd.DataFrame:
    catalog_path = Path(path)
    if not catalog_path.exists():
        return pd.DataFrame()
    return pd.read_csv(catalog_path)


@st.cache_data(show_spinner=False)
def load_pdf_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False)
def load_pdf_page_count(path: str) -> int:
    with pymupdf.open(path) as document:
        return document.page_count


@st.cache_data(show_spinner=False)
def render_pdf_page(path: str, page_index: int, zoom: float) -> bytes:
    with pymupdf.open(path) as document:
        page = document.load_page(page_index)
        matrix = pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")


def render_trace(trace: list[Any]) -> None:
    with st.expander("工具调用记录", expanded=False):
        if not trace:
            st.info("暂无工具调用记录。")
            return
        for item in trace:
            st.markdown(f"**Step {item.step}: {item.tool_name}**  ·  {item.elapsed_ms:.2f} ms")
            cols = st.columns(2)
            with cols[0]:
                st.caption("Input")
                st.code(_safe_json(item.tool_input), language="json")
            with cols[1]:
                st.caption("Output")
                st.code(_safe_json(item.tool_output), language="json")


def render_agent_details(result: AgentRunResult, expanded: bool = False, divider: bool = True) -> None:
    if divider:
        st.divider()
    with st.expander("Agent 执行过程", expanded=expanded):
        st.caption("这里展示参数抽取、标准器推荐、规则校验、标准引用和工具调用记录，用于解释答案来源。")

        if result.parameters:
            st.subheader("识别参数")
            st.json(result.parameters.model_dump())

        if result.recommendations:
            st.subheader("推荐标准器")
            rows = []
            for rec in result.recommendations:
                inst = rec.instrument
                rows.append(
                    {
                        "id": inst.id,
                        "name": inst.name,
                        "type": inst.type,
                        "model": inst.model,
                        "range": f"{inst.range_min}～{inst.range_max} {inst.unit}",
                        "accuracy_class": inst.accuracy_class,
                        "score": rec.score,
                        "reason": rec.reason,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("规则校验")
            check_rows = []
            for rec in result.recommendations:
                for check in rec.validation.checks:
                    check_rows.append(
                        {
                            "instrument_id": rec.instrument.id,
                            "rule": check.rule,
                            "status": "通过" if check.passed else "未通过",
                            "reason": check.reason,
                        }
                    )
            st.dataframe(pd.DataFrame(check_rows), use_container_width=True, hide_index=True)

        if result.citations:
            st.subheader("标准引用")
            citation_rows = [item.model_dump() for item in result.citations]
            st.dataframe(pd.DataFrame(citation_rows), use_container_width=True, hide_index=True)

        render_trace(result.trace)


def render_answer_tab() -> None:
    default_question = "检定 0～1.6 MPa、1.6 级压力表应该选择什么标准器？"
    question = st.text_area("问题", value=default_question, height=86)
    run_clicked = st.button("生成建议", type="primary", use_container_width=False)

    if not run_clicked:
        return

    if not question.strip():
        st.warning("请输入问题。")
        return

    try:
        with st.spinner("正在检索标准并校验标准器..."):
            result = MetrologyAgentRunner(settings).run(question.strip())
    except Exception as exc:
        logger.exception("Agent run failed: %s", exc)
        st.error("Agent 执行失败，请检查配置、知识库和标准器 CSV。")
        return

    if result.error:
        st.warning(f"执行过程中发生异常，已给出可用结果：{result.error}")

    st.subheader("答案")
    st.markdown(result.answer)
    render_agent_details(result)


def _init_chat_state() -> None:
    st.session_state.setdefault("metrology_chat_messages", [])
    st.session_state.setdefault("metrology_chat_state", {})


def _chat_history() -> list[dict[str, str]]:
    history = []
    for item in st.session_state.get("metrology_chat_messages", []):
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history


def render_chat_tab() -> None:
    _init_chat_state()

    action_cols = st.columns([1, 1, 5])
    if action_cols[0].button("清空对话"):
        st.session_state["metrology_chat_messages"] = []
        st.session_state["metrology_chat_state"] = {}
        st.rerun()
    with action_cols[1]:
        with st.expander("上下文", expanded=False):
            st.json(st.session_state.get("metrology_chat_state") or {})

    for message in st.session_state["metrology_chat_messages"]:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant" and message.get("result"):
                render_agent_details(
                    AgentRunResult.model_validate(message["result"]),
                    expanded=False,
                    divider=True,
                )

    prompt = st.chat_input("请输入问题或需求...")
    if not prompt:
        return

    history = _chat_history()
    st.session_state["metrology_chat_messages"].append({"role": "user", "content": prompt})
    try:
        with st.spinner("正在结合上下文生成建议..."):
            result = MetrologyAgentRunner(settings).run(
                prompt,
                chat_history=history,
                structured_state=st.session_state.get("metrology_chat_state") or {},
            )
    except Exception as exc:
        logger.exception("Agent chat run failed: %s", exc)
        st.session_state["metrology_chat_messages"].append(
            {"role": "assistant", "content": "执行失败，请检查配置、知识库和标准器 CSV。"}
        )
        st.rerun()

    if result.structured_state:
        st.session_state["metrology_chat_state"] = result.structured_state.model_dump()
    st.session_state["metrology_chat_messages"].append(
        {
            "role": "assistant",
            "content": result.answer,
            "result": result.model_dump(),
        }
    )
    st.rerun()


def render_pdf_preview() -> None:
    pdf_files = sorted(settings.pdf_dir.glob("*.pdf"))
    st.subheader("PDF 预览")
    if not pdf_files:
        st.info("data/pdf/ 为空，上传 PDF 后可在这里预览。")
        return

    file_options = [pdf_path.name for pdf_path in pdf_files]
    selected_name = st.selectbox("选择 PDF", file_options)
    selected_path = next(pdf_path for pdf_path in pdf_files if pdf_path.name == selected_name)

    try:
        pdf_bytes = load_pdf_bytes(str(selected_path))
    except OSError as exc:
        logger.exception("PDF preview failed: %s", exc)
        st.error("PDF 读取失败，请检查文件权限。")
        return

    size_mb = len(pdf_bytes) / 1024 / 1024
    cols = st.columns([3, 1])
    cols[0].caption(f"{selected_path} · {size_mb:.2f} MB")
    cols[1].download_button(
        "下载 PDF",
        data=pdf_bytes,
        file_name=selected_path.name,
        mime="application/pdf",
        use_container_width=True,
    )

    try:
        page_count = load_pdf_page_count(str(selected_path))
    except Exception as exc:
        logger.exception("PDF page count failed: %s", exc)
        st.error("PDF 解析失败，请检查文件是否损坏。")
        return

    try:
        with st.spinner("正在渲染 PDF 预览..."):
            page_images = [
                render_pdf_page(str(selected_path), page_index, PDF_PREVIEW_ZOOM)
                for page_index in range(page_count)
            ]
    except Exception as exc:
        logger.exception("PDF render failed: %s", exc)
        st.error("PDF 页面渲染失败，请检查文件是否损坏。")
        return

    st.caption(f"共 {page_count} 页 · 预览清晰度 {PDF_PREVIEW_ZOOM}x · 在下方窗口内滚轮翻页")
    with st.container(height=PDF_PREVIEW_HEIGHT, border=True):
        for page_number, page_image in enumerate(page_images, start=1):
            st.caption(f"第 {page_number} 页")
            st.image(page_image, use_container_width=True)


def render_knowledge_tab() -> None:
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.markdown_dir.mkdir(parents=True, exist_ok=True)

    pdf_count = len(list(settings.pdf_dir.glob("*.pdf")))
    markdown_count = len(list(settings.markdown_dir.glob("*.md")))
    try:
        chunk_count = count_markdown_chunks(settings)
    except Exception:
        chunk_count = 0
    chroma_count = get_chroma_count(settings)

    cols = st.columns(4)
    cols[0].metric("PDF 数量", pdf_count)
    cols[1].metric("Markdown 数量", markdown_count)
    cols[2].metric("Chunk 数量", chunk_count)
    cols[3].metric("Chroma Collection 数量", chroma_count)

    uploaded_files = st.file_uploader(
        "上传 PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("保存文件"):
        saved = 0
        for uploaded_file in uploaded_files:
            target_path = settings.pdf_dir / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())
            saved += 1
        st.success(f"已保存 {saved} 个 PDF。")

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("转换 PDF", use_container_width=True):
            try:
                with st.spinner("正在转换 PDF..."):
                    converted = convert_pdf_directory(settings.pdf_dir, settings.markdown_dir, settings)
                if converted:
                    st.success(f"已转换 {len(converted)} 个 PDF。")
                else:
                    st.info("data/pdf/ 为空，可先上传 PDF；当前仍可使用 demo_standard.md 演示。")
            except Exception as exc:
                logger.exception("PDF conversion failed: %s", exc)
                st.error("PDF 转换失败，请检查 PDF 是否可读取。")

    with action_cols[1]:
        if st.button("重建知识库", use_container_width=True):
            try:
                with st.spinner("正在构建 Chroma 知识库..."):
                    _, chunks = build_vector_store(rebuild=True, app_settings=settings)
                st.success(f"知识库构建完成，共写入 {chunks} 个 chunk。")
            except Exception as exc:
                logger.exception("Vector store build failed: %s", exc)
                st.error("知识库构建失败，请检查 Embedding 配置或文档内容。")

    render_pdf_preview()


def render_catalog_tab() -> None:
    df = load_catalog(str(settings.instruments_path))
    if df.empty:
        st.warning(f"未找到标准器目录：{settings.instruments_path}")
        return
    if settings.instruments_path.name == "instruments_demo.csv":
        st.caption("当前使用公开 Demo 标准器目录；本地存在 data/instruments.csv 时会优先读取本地目录。")

    filter_cols = st.columns(4)
    type_options = ["全部"] + sorted(str(item) for item in df["type"].dropna().unique())
    selected_type = filter_cols[0].selectbox("设备类型", type_options)
    unit_options = ["全部"] + sorted(str(item) for item in df["unit"].dropna().unique() if str(item).strip())
    selected_unit = filter_cols[1].selectbox("单位", unit_options)
    range_max = filter_cols[2].number_input("最小覆盖上限", min_value=0.0, value=0.0, step=0.1)
    accuracy_max = filter_cols[3].number_input("最大准确度数值", min_value=0.0, value=0.0, step=0.01)

    filtered = df.copy()
    if selected_type != "全部":
        filtered = filtered[filtered["type"].astype(str) == selected_type]
    if selected_unit != "全部":
        filtered = filtered[filtered["unit"].astype(str) == selected_unit]
    if range_max > 0 and "range_max" in filtered:
        filtered = filtered[pd.to_numeric(filtered["range_max"], errors="coerce") >= range_max]
    if accuracy_max > 0 and "accuracy_class" in filtered:
        filtered = filtered[pd.to_numeric(filtered["accuracy_class"], errors="coerce") <= accuracy_max]

    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_evaluation_tab() -> None:
    if st.button("运行评估", type="primary"):
        try:
            with st.spinner("正在运行测试集..."):
                summary = run_evaluation(settings)
        except Exception as exc:
            logger.exception("Evaluation failed: %s", exc)
            st.error("Evaluation 运行失败，请检查 evaluation_cases.json、知识库和标准器 CSV。")
            return

        recommendation_rate = (
            sum(item.recommendation_success for item in summary.case_results) / summary.total_cases
            if summary.total_cases
            else 0.0
        )
        citation_rate = (
            sum(item.citation_success for item in summary.case_results) / summary.total_cases
            if summary.total_cases
            else 0.0
        )
        golden_hit_rate = (
            sum(item.expected_device_hit for item in summary.case_results) / summary.total_cases
            if summary.total_cases
            else 0.0
        )

        cols = st.columns(5)
        cols[0].metric("测试集总数", summary.total_cases)
        cols[1].metric("工具调用成功率", f"{summary.tool_call_success_rate:.1%}")
        cols[2].metric("端到端成功率", f"{summary.end_to_end_success_rate:.1%}")
        cols[3].metric("平均工具调用次数", f"{summary.average_tool_calls:.2f}")
        cols[4].metric("平均响应时间", f"{summary.average_response_time_ms:.0f} ms")

        diagnostic_cols = st.columns(3)
        diagnostic_cols[0].metric("规则推荐成功率", f"{recommendation_rate:.1%}")
        diagnostic_cols[1].metric("引用成功率", f"{citation_rate:.1%}")
        diagnostic_cols[2].metric("Golden ID 命中率", f"{golden_hit_rate:.1%}")

        column_order = [
            "id",
            "question",
            "tool_call_success",
            "end_to_end_success",
            "recommendation_success",
            "citation_success",
            "expected_device_hit",
            "failure_reason",
            "expected_tools",
            "actual_tools",
            "response_time_ms",
            "error",
        ]
        rows = [item.model_dump() for item in summary.case_results]
        df = pd.DataFrame(rows)
        ordered_columns = [column for column in column_order if column in df.columns]
        st.dataframe(df[ordered_columns], use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    render_header()

    tabs = st.tabs(["单轮问答", "多轮问答", "知识库", "标准器", "评估"])
    with tabs[0]:
        render_answer_tab()
    with tabs[1]:
        render_chat_tab()
    with tabs[2]:
        render_knowledge_tab()
    with tabs[3]:
        render_catalog_tab()
    with tabs[4]:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
