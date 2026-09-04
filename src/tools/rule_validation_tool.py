from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.rules.validator import validate_instrument


class ValidateInstrumentInput(BaseModel):
    parameters: dict[str, Any] = Field(..., description="被检仪器参数 JSON")
    instrument: dict[str, Any] = Field(..., description="候选标准器 JSON")


def validate_instrument_tool(parameters: dict[str, Any], instrument: dict[str, Any]) -> dict[str, Any]:
    return validate_instrument(parameters, instrument).model_dump()


def get_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="validate_instrument",
        description="用 Python 规则引擎校验候选标准器是否满足量程覆盖和准确度比例等规则。",
        func=validate_instrument_tool,
        args_schema=ValidateInstrumentInput,
    )

