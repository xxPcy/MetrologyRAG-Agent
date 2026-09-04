from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from config.settings import AppSettings, settings
from src.models.schemas import InstrumentParameters, InstrumentRecord
from src.rules.validator import canonicalize_instrument_type, convert_value, load_rules


class QueryInstrumentCatalogInput(BaseModel):
    parameters: dict[str, Any] = Field(..., description="extract_parameters 返回的被检仪器参数 JSON")
    limit: int = Field(default=10, ge=1, le=50, description="最多返回候选数量")


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_instrument_catalog(path: Path | None = None) -> list[InstrumentRecord]:
    catalog_path = path or settings.instruments_path
    if not catalog_path.exists():
        raise FileNotFoundError(f"标准器 CSV 不存在: {catalog_path}")

    records: list[InstrumentRecord] = []
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["range_min"] = _to_float(row.get("range_min"))
            row["range_max"] = _to_float(row.get("range_max"))
            row["accuracy_class"] = _to_float(row.get("accuracy_class"))
            records.append(InstrumentRecord.model_validate(row))
    return records


def _range_covers(candidate: InstrumentRecord, params: InstrumentParameters) -> bool:
    if (
        params.range_min is None
        or params.range_max is None
        or not params.unit
        or candidate.range_min is None
        or candidate.range_max is None
        or not candidate.unit
    ):
        return True
    try:
        candidate_min = convert_value(candidate.range_min, candidate.unit, params.unit)
        candidate_max = convert_value(candidate.range_max, candidate.unit, params.unit)
    except ValueError:
        return False
    return candidate_min <= params.range_min and candidate_max >= params.range_max


def _sort_key(candidate: InstrumentRecord, params: InstrumentParameters) -> tuple[int, float, float, str]:
    accuracy = candidate.accuracy_class if candidate.accuracy_class is not None else 999999.0
    if candidate.range_max is None or params.range_max is None or not candidate.unit or not params.unit:
        range_margin = 999999.0
    else:
        try:
            range_margin = abs(convert_value(candidate.range_max, candidate.unit, params.unit) - params.range_max)
        except ValueError:
            range_margin = 999999.0
    source_priority = 0 if candidate.source == "demo" else 1
    return (source_priority, range_margin, accuracy, candidate.id)


def query_instrument_catalog(
    parameters: dict[str, Any],
    limit: int = 10,
    app_settings: AppSettings = settings,
) -> list[dict[str, Any]]:
    """Query candidates by instrument type, range, and basic accuracy fields."""
    params = InstrumentParameters.model_validate(parameters)
    rules = load_rules(str(app_settings.rules_path))
    rule_type = canonicalize_instrument_type(params.instrument_type, rules)
    allowed_types = set(rules.get(rule_type or "", {}).get("standard_instrument_types") or [])

    candidates: list[InstrumentRecord] = []
    for record in load_instrument_catalog(app_settings.instruments_path):
        if allowed_types and record.type not in allowed_types:
            continue
        if not _range_covers(record, params):
            continue
        if (
            params.accuracy_class is not None
            and record.accuracy_class is not None
            and record.accuracy_class > params.accuracy_class
        ):
            continue
        candidates.append(record)

    candidates.sort(key=lambda candidate: _sort_key(candidate, params))
    return [candidate.model_dump() for candidate in candidates[:limit]]


def get_tool(app_settings: AppSettings = settings) -> StructuredTool:
    def _run(parameters: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
        return query_instrument_catalog(parameters, limit, app_settings)

    return StructuredTool.from_function(
        name="query_instrument_catalog",
        description="根据被检仪器类型、量程和准确度等级，从标准器目录 CSV 查询候选设备。",
        func=_run,
        args_schema=QueryInstrumentCatalogInput,
    )
