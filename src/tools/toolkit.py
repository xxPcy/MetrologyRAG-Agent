from __future__ import annotations

from langchain_core.tools import BaseTool

from config.settings import AppSettings, settings
from src.tools import (
    instrument_query_tool,
    parameter_extract_tool,
    rag_search_tool,
    recommendation_tool,
    rule_validation_tool,
)


def build_tools(app_settings: AppSettings = settings) -> list[BaseTool]:
    return [
        rag_search_tool.get_tool(app_settings),
        parameter_extract_tool.get_tool(),
        instrument_query_tool.get_tool(app_settings),
        rule_validation_tool.get_tool(),
        recommendation_tool.get_tool(),
    ]

