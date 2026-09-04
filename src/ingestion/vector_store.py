from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import AppSettings, settings
from src.ingestion.chunker import chunk_markdown_files
from src.ingestion.embeddings import get_embedding_model
from src.ingestion.pdf_to_markdown import convert_pdf_directory
from src.models.schemas import SearchResult
from src.utils.logger import get_logger


logger = get_logger(__name__)


def _markdown_sources(app_settings: AppSettings) -> list[Path]:
    markdown_files = sorted(app_settings.markdown_dir.glob("*.md"))
    if markdown_files:
        return markdown_files
    if app_settings.demo_standard_path.exists():
        return [app_settings.demo_standard_path]
    return []


def build_vector_store(
    rebuild: bool = True,
    app_settings: AppSettings = settings,
    convert_pdfs: bool = True,
) -> tuple[Chroma, int]:
    """Convert PDFs if present, chunk Markdown, and persist a Chroma collection."""
    app_settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    app_settings.markdown_dir.mkdir(parents=True, exist_ok=True)
    app_settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    if convert_pdfs:
        convert_pdf_directory(app_settings.pdf_dir, app_settings.markdown_dir, app_settings)
    markdown_files = _markdown_sources(app_settings)
    documents = chunk_markdown_files(markdown_files, app_settings) if markdown_files else []
    if not documents:
        raise RuntimeError("没有可索引的 Markdown 文档，请上传 PDF 或保留 demo_standard.md。")

    if rebuild and app_settings.chroma_dir.exists():
        shutil.rmtree(app_settings.chroma_dir)
        app_settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_model(app_settings),
        collection_name=app_settings.collection_name,
        persist_directory=str(app_settings.chroma_dir),
    )
    logger.info("Built vector store with %s chunks.", len(documents))
    return vector_store, len(documents)


def load_vector_store(
    create_if_missing: bool = True,
    app_settings: AppSettings = settings,
) -> Chroma:
    if not app_settings.chroma_dir.exists() or not any(app_settings.chroma_dir.iterdir()):
        if not create_if_missing:
            raise FileNotFoundError("Chroma 数据库不存在，请先构建知识库。")
        vector_store, _ = build_vector_store(rebuild=True, app_settings=app_settings)
        return vector_store

    return Chroma(
        collection_name=app_settings.collection_name,
        embedding_function=get_embedding_model(app_settings),
        persist_directory=str(app_settings.chroma_dir),
    )


def similarity_search(
    query: str,
    top_k: int | None = None,
    app_settings: AppSettings = settings,
) -> list[SearchResult]:
    k = top_k or app_settings.retrieval_top_k
    vector_store = load_vector_store(create_if_missing=True, app_settings=app_settings)
    try:
        docs_and_scores: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
            query, k=k
        )
    except Exception as exc:
        logger.exception("Similarity search failed: %s", exc)
        raise RuntimeError("RAG 检索失败，请检查 Embedding 配置或知识库状态。") from exc

    results: list[SearchResult] = []
    for document, raw_score in docs_and_scores:
        score = 1.0 / (1.0 + float(raw_score)) if raw_score is not None else None
        metadata = document.metadata
        results.append(
            SearchResult(
                content=document.page_content,
                source=str(metadata.get("source", "")),
                page=metadata.get("page"),
                section=str(metadata.get("section", "")),
                score=round(score, 4) if score is not None else None,
            )
        )
    return results


def count_markdown_chunks(app_settings: AppSettings = settings) -> int:
    markdown_files = _markdown_sources(app_settings)
    if not markdown_files:
        return 0
    return len(chunk_markdown_files(markdown_files, app_settings))


def get_chroma_count(app_settings: AppSettings = settings) -> int:
    if not app_settings.chroma_dir.exists() or not any(app_settings.chroma_dir.iterdir()):
        return 0
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(app_settings.chroma_dir))
        collection = client.get_or_create_collection(app_settings.collection_name)
        return collection.count()
    except Exception as exc:
        logger.warning("Unable to read Chroma count: %s", exc)
        return 0
