from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config.settings import AppSettings, settings


PAGE_MARKER_RE = re.compile(
    r"<!--\s*source:\s*(?P<source>.*?)\s*\|\s*page:\s*(?P<page>\d+)\s*-->",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MarkdownPageBlock:
    source: str
    page: int | None
    text: str


def _page_blocks(markdown_path: Path) -> list[MarkdownPageBlock]:
    text = markdown_path.read_text(encoding="utf-8")
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        return [
            MarkdownPageBlock(source=markdown_path.name, page=1, text=text),
        ]

    blocks: list[MarkdownPageBlock] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(
            MarkdownPageBlock(
                source=match.group("source").strip(),
                page=int(match.group("page")),
                text=text[match.end() : next_start].strip(),
            )
        )
    return blocks


def _section_from_metadata(metadata: dict[str, str]) -> str:
    for key in ("h4", "h3", "h2", "h1"):
        value = metadata.get(key)
        if value and not value.lower().startswith("page "):
            return value
    return metadata.get("h2", "") or metadata.get("h1", "")


def chunk_markdown_file(
    markdown_path: Path,
    app_settings: AppSettings = settings,
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=app_settings.chunk_size,
        chunk_overlap=app_settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "；", ";", " ", ""],
    )

    header_documents: list[Document] = []
    for block in _page_blocks(markdown_path):
        split_docs = header_splitter.split_text(block.text)
        for doc in split_docs:
            section = _section_from_metadata(doc.metadata)
            doc.metadata.update(
                {
                    "source": block.source,
                    "page": block.page,
                    "section": section,
                    "markdown_file": markdown_path.name,
                }
            )
            header_documents.append(doc)

    chunks = text_splitter.split_documents(header_documents)
    for index, doc in enumerate(chunks, start=1):
        page = doc.metadata.get("page") or "na"
        doc.metadata["chunk_id"] = f"{markdown_path.stem}:p{page}:c{index}"
    return chunks


def chunk_markdown_files(
    markdown_paths: list[Path],
    app_settings: AppSettings = settings,
) -> list[Document]:
    chunks: list[Document] = []
    for markdown_path in markdown_paths:
        chunks.extend(chunk_markdown_file(markdown_path, app_settings))
    return chunks

