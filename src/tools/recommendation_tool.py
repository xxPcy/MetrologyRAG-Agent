from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.models.schemas import InstrumentParameters, InstrumentRecord, Recommendation
from src.rules.validator import validate_instrument


class RecommendInstrumentsInput(BaseModel):
    parameters: dict[str, Any] = Field(..., description="被检仪器参数 JSON")
    candidates: list[dict[str, Any]] = Field(..., description="query_instrument_catalog 返回的候选标准器列表")
    limit: int = Field(default=5, ge=1, le=20, description="最多返回推荐数量")


def _recommendation_score(instrument: InstrumentRecord, params: InstrumentParameters) -> float:
    accuracy = instrument.accuracy_class if instrument.accuracy_class is not None else 999.0
    range_width = (
        abs((instrument.range_max or 0.0) - (instrument.range_min or 0.0))
        if instrument.range_min is not None and instrument.range_max is not None
        else 999.0
    )
    target_width = (
        abs(params.range_max - params.range_min)
        if params.range_min is not None and params.range_max is not None
        else 1.0
    )
    range_fit = 1.0 / (1.0 + abs(range_width - target_width))
    accuracy_fit = 1.0 / (1.0 + accuracy)
    return round(range_fit + accuracy_fit, 4)


def recommend_instruments(
    parameters: dict[str, Any],
    candidates: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    params = InstrumentParameters.model_validate(parameters)
    recommendations: list[Recommendation] = []
    for item in candidates:
        instrument = InstrumentRecord.model_validate(item)
        validation = validate_instrument(params, instrument)
        if not validation.passed:
            continue
        passed_reasons = [check.reason for check in validation.checks if check.passed]
        recommendations.append(
            Recommendation(
                instrument=instrument,
                validation=validation,
                reason="；".join(passed_reasons),
                score=_recommendation_score(instrument, params),
            )
        )

    recommendations.sort(key=lambda rec: (-rec.score, rec.instrument.id))
    return [rec.model_dump() for rec in recommendations[:limit]]


def get_tool() -> StructuredTool:
    return StructuredTool.from_function(
        name="recommend_instruments",
        description="对候选标准器逐个执行规则校验、过滤、排序并返回推荐结果。",
        func=recommend_instruments,
        args_schema=RecommendInstrumentsInput,
    )

