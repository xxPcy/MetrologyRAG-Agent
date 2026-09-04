from __future__ import annotations

import re
from pathlib import Path

from config.settings import AppSettings, settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", path.stem).strip("_")


def _detect_section(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^(\d+(\.\d+)*|[一二三四五六七八九十]+[、.])\s*[\u4e00-\u9fffA-Za-z]", line):
            return line[:80]
    return ""


def _extract_page_with_pymupdf4llm(pdf_path: Path, page_index: int) -> str | None:
    try:
        import pymupdf4llm

        markdown = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_index])
        if isinstance(markdown, str) and markdown.strip():
            return markdown
    except Exception as exc:
        logger.debug("pymupdf4llm page extraction skipped: %s", exc)
    return None


def _extract_page_with_pymupdf(page: object) -> str:
    text = page.get_text("text")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line.strip())


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path) -> Path:
    """Convert one PDF to page-marked Markdown with source and page metadata."""
    try:
        import pymupdf as fitz
    except Exception:
        try:
            import fitz
        except Exception as exc:
            raise RuntimeError("PyMuPDF is required to convert PDF files.") from exc

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_stem(pdf_path)}.md"

    parts: list[str] = [f"# {pdf_path.stem}", ""]
    with fitz.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            text = _extract_page_with_pymupdf4llm(pdf_path, page_index)
            if not text:
                text = _extract_page_with_pymupdf(document.load_page(page_index))
            section = _detect_section(text)
            parts.append(f"<!-- source: {pdf_path.name} | page: {page_number} -->")
            parts.append(f"## Page {page_number}")
            if section:
                parts.append(f"### {section}")
            parts.append(text.strip() or "_本页未提取到文本。_")
            parts.append("")

    output_path.write_text("\n".join(parts), encoding="utf-8")
    logger.info("Converted PDF to Markdown: %s -> %s", pdf_path.name, output_path)
    return output_path


def convert_pdf_directory(
    pdf_dir: Path | None = None,
    markdown_dir: Path | None = None,
    app_settings: AppSettings = settings,
) -> list[Path]:
    source_dir = pdf_dir or app_settings.pdf_dir
    target_dir = markdown_dir or app_settings.markdown_dir
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(source_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s", source_dir)
        return []

    converted: list[Path] = []
    for pdf_path in pdf_files:
        converted.append(convert_pdf_to_markdown(pdf_path, target_dir))
    return converted
