from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.settings import AppSettings, settings
from src.rag.retriever import retrieve_standard_chunks


class RagSearchInput(BaseModel):
    query: str = Field(..., description="检索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="返回条数")


def search_metrology_standard(
    query: str,
    top_k: int = 5,
    app_settings: AppSettings = settings,
) -> list[dict[str, Any]]:
    results = retrieve_standard_chunks(query=query, top_k=top_k, app_settings=app_settings)
    return [result.model_dump() for result in results]


def get_tool(app_settings: AppSettings = settings) -> StructuredTool:
    def _run(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return search_metrology_standard(query, top_k, app_settings)

    return StructuredTool.from_function(
        name="search_metrology_standard",
        description="从 Chroma 知识库检索国家计量标准、JJG 检定规程、JJF 校准规范相关片段。",
        func=_run,
        args_schema=RagSearchInput,
    )

