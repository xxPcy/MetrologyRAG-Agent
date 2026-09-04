from __future__ import annotations

from config.settings import AppSettings, settings
from src.ingestion.vector_store import similarity_search
from src.models.schemas import SearchResult


def retrieve_standard_chunks(
    query: str,
    top_k: int | None = None,
    app_settings: AppSettings = settings,
) -> list[SearchResult]:
    return similarity_search(query=query, top_k=top_k, app_settings=app_settings)

