from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.models.schemas import InstrumentParameters
from src.rules.validator import canonicalize_instrument_type, normalize_unit


RANGE_RE = re.compile(
    r"(?P<min>-?\d+(?:\.\d+)?)\s*(?:~|～|至|到|-)\s*"
    r"(?P<max>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>MPa|Mpa|mpa|kPa|KPa|Pa|bar|mm|cm|m|μm|µm|um|℃|°C|N|kN|kg|g|L|mL|ml)",
    re.IGNORECASE,
)
ACCURACY_CLASS_RE = re.compile(r"(?<!\d)(?P<value>\d+(?:\.\d+)?)\s*(?:级|class)", re.IGNORECASE)
MPE_RE = re.compile(r"MPE\s*[:：]?\s*[±+/-]*\s*(?P<value>\d+(?:\.\d+)?)", re.IGNORECASE)


TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("pressure_gauge", ["压力真空表", "一般压力表", "压力表", "真空表"]),
    ("digital_pressure_gauge", ["数字压力计", "数字压力表"]),
    ("caliper", ["通用卡尺", "游标卡尺", "数显卡尺", "卡尺"]),
    ("dial_indicator", ["百分表", "千分表", "指示表"]),
    ("micrometer", ["外径千分尺", "内径千分尺", "千分尺"]),
    ("thermometer", ["温度计", "热电偶", "铂电阻"]),
    ("weighing_instrument", ["电子秤", "天平", "秤"]),
    ("volumetric_instrument", ["容量瓶", "量筒", "移液器", "玻璃量器"]),
]


class ExtractParametersInput(BaseModel):
    text: str = Field(..., description="用户问题或标准条款文本")
    context: str | None = Field(default=None, description="可选检索上下文")


def _detect_type(text: str) -> str | None:
    for canonical_type, keywords in TYPE_PATTERNS:
        if any(keyword in text for keyword in keywords):
            return canonicalize_instrument_type(canonical_type)
    return None


def extract_parameters(text: str, context: str | None = None) -> dict[str, Any]:
    """Extract metrology parameters with deterministic regex and Pydantic output."""
    range_match = RANGE_RE.search(text)
    accuracy_match = ACCURACY_CLASS_RE.search(text)
    mpe_match = MPE_RE.search(text)

    params = InstrumentParameters(
        instrument_type=_detect_type(text) or _detect_type(context or ""),
        range_min=float(range_match.group("min")) if range_match else None,
        range_max=float(range_match.group("max")) if range_match else None,
        unit=normalize_unit(range_match.group("unit")) if range_match else None,
        accuracy_class=(
            float(accuracy_match.group("value"))
            if accuracy_match
            else float(mpe_match.group("value"))
            if mpe_match
            else None
        ),
        accuracy_text=accuracy_match.group(0) if accuracy_match else mpe_match.group(0) if mpe_match else None,
        raw_text=text,
    )
    return params.model_dump()


def get_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="extract_parameters",
        description="从用户需求或标准条款中抽取被检仪器类型、量程、单位和准确度等级，返回严格 JSON。",
        func=extract_parameters,
        args_schema=ExtractParametersInput,
    )
