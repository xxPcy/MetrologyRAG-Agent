from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from config.settings import settings
from src.models.schemas import InstrumentParameters, InstrumentRecord, ValidationCheck, ValidationResult


UNIT_FACTORS: dict[str, tuple[str, float]] = {
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1_000.0),
    "mpa": ("pressure", 1_000_000.0),
    "bar": ("pressure", 100_000.0),
    "m": ("length", 1000.0),
    "cm": ("length", 10.0),
    "mm": ("length", 1.0),
    "um": ("length", 0.001),
    "μm": ("length", 0.001),
    "µm": ("length", 0.001),
    "nm": ("length", 0.000001),
    "kg": ("mass", 1000.0),
    "g": ("mass", 1.0),
    "mg": ("mass", 0.001),
    "kn": ("force", 1000.0),
    "n": ("force", 1.0),
    "l": ("volume", 1000.0),
    "ml": ("volume", 1.0),
    "℃": ("temperature", 1.0),
    "°c": ("temperature", 1.0),
    "c": ("temperature", 1.0),
}


STANDARD_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("pressure_calibrator", ["压力校验仪", "压力校准器", "压力校准仪"]),
    ("digital_pressure_gauge", ["数字压力计", "数字压力表"]),
    ("precision_pressure_gauge", ["精密压力表", "精密真空表"]),
    ("caliper_checker", ["卡尺检定器", "卡尺量具校准器"]),
    ("dial_indicator_tester", ["百分表检定仪", "千分表检定仪", "指示表检定仪"]),
    ("micrometer_checker", ["千分尺检定器", "千分尺校准器"]),
    ("gauge_block", ["量块", "标准量块"]),
    ("thermometer", ["温度计", "铂电阻", "热电偶"]),
    ("balance_weight", ["砝码", "标准砝码"]),
    ("volumetric_standard", ["容量瓶", "量筒", "移液器", "玻璃量器"]),
]


def normalize_unit(unit: str | None) -> str:
    if not unit:
        return ""
    text = unit.strip().replace("μ", "μ").replace("－", "-")
    aliases = {
        "Mpa": "MPa",
        "mpa": "MPa",
        "KPa": "kPa",
        "kpa": "kPa",
        "UM": "μm",
        "um": "μm",
        "uM": "μm",
        "ml": "mL",
        "ML": "mL",
        "℃": "℃",
        "°C": "℃",
    }
    return aliases.get(text, text)


def _unit_key(unit: str | None) -> str:
    return normalize_unit(unit).lower().replace("μ", "μ").replace("µ", "µ")


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    from_key = _unit_key(from_unit)
    to_key = _unit_key(to_unit)
    if from_key == to_key:
        return value
    if from_key not in UNIT_FACTORS or to_key not in UNIT_FACTORS:
        raise ValueError(f"单位不可换算: {from_unit} -> {to_unit}")
    from_category, from_factor = UNIT_FACTORS[from_key]
    to_category, to_factor = UNIT_FACTORS[to_key]
    if from_category != to_category:
        raise ValueError(f"单位量纲不一致: {from_unit} -> {to_unit}")
    return value * from_factor / to_factor


@lru_cache(maxsize=4)
def load_rules(rules_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(rules_path or settings.rules_path)
    if not path.exists():
        raise FileNotFoundError(f"规则配置不存在: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError("规则配置必须是 YAML mapping。")
    return data


def canonicalize_instrument_type(text: str | None, rules: dict[str, Any] | None = None) -> str | None:
    if not text:
        return None
    normalized = str(text).strip()
    if not normalized:
        return None

    lower = normalized.lower()
    active_rules = rules or load_rules()
    if lower in active_rules:
        return lower

    for canonical, keywords in STANDARD_TYPE_KEYWORDS:
        if canonical == lower or any(keyword in normalized for keyword in keywords):
            return canonical

    for rule_type, spec in active_rules.items():
        aliases = spec.get("aliases", [])
        if rule_type == lower or any(alias in normalized for alias in aliases):
            return rule_type
    return lower


def validate_type(
    parameters: InstrumentParameters,
    instrument: InstrumentRecord,
    rule_spec: dict[str, Any],
) -> ValidationCheck:
    allowed_types = set(rule_spec.get("standard_instrument_types") or [])
    if not allowed_types:
        return ValidationCheck(rule="type", passed=True, reason="规则未限制标准器类型。")
    passed = instrument.type in allowed_types
    return ValidationCheck(
        rule="type",
        passed=passed,
        reason=(
            "标准器类型在配置允许范围内。"
            if passed
            else f"标准器类型 {instrument.type} 不在允许范围: {sorted(allowed_types)}。"
        ),
        actual=instrument.type,
        expected=sorted(allowed_types),
    )


def validate_range(
    parameters: InstrumentParameters,
    instrument: InstrumentRecord,
    rule_spec: dict[str, Any],
) -> ValidationCheck:
    range_rule = rule_spec.get("range", {})
    if not range_rule.get("coverage_required", False):
        return ValidationCheck(rule="range", passed=True, reason="规则未要求量程覆盖。")

    if (
        parameters.range_min is None
        or parameters.range_max is None
        or not parameters.unit
        or instrument.range_min is None
        or instrument.range_max is None
        or not instrument.unit
    ):
        return ValidationCheck(
            rule="range",
            passed=False,
            reason="被检仪器或标准器缺少可计算的量程参数。",
        )

    try:
        candidate_min = convert_value(instrument.range_min, instrument.unit, parameters.unit)
        candidate_max = convert_value(instrument.range_max, instrument.unit, parameters.unit)
    except ValueError as exc:
        return ValidationCheck(rule="range", passed=False, reason=str(exc))

    passed = candidate_min <= parameters.range_min and candidate_max >= parameters.range_max
    return ValidationCheck(
        rule="range",
        passed=passed,
        reason=(
            "标准器量程覆盖被检仪器量程。"
            if passed
            else "标准器量程未覆盖被检仪器量程。"
        ),
        actual={
            "standard_range": [candidate_min, candidate_max],
            "unit": parameters.unit,
        },
        expected={
            "dut_range": [parameters.range_min, parameters.range_max],
            "unit": parameters.unit,
        },
    )


def validate_accuracy(
    parameters: InstrumentParameters,
    instrument: InstrumentRecord,
    rule_spec: dict[str, Any],
) -> ValidationCheck:
    accuracy_rule = rule_spec.get("accuracy", {})
    ratio = float(accuracy_rule.get("ratio") or 0)
    if ratio <= 0:
        return ValidationCheck(rule="accuracy", passed=True, reason="规则未配置准确度比例。")

    if parameters.accuracy_class is None or instrument.accuracy_class is None:
        return ValidationCheck(
            rule="accuracy",
            passed=False,
            reason="被检仪器或标准器缺少可计算的准确度等级。",
        )

    expected_max = parameters.accuracy_class / ratio
    if accuracy_rule.get("lower_is_better", True):
        passed = instrument.accuracy_class <= expected_max
    else:
        passed = instrument.accuracy_class >= expected_max

    return ValidationCheck(
        rule="accuracy",
        passed=passed,
        reason=(
            f"准确度满足配置的 {ratio:g}:1 比例规则。"
            if passed
            else f"准确度不满足配置的 {ratio:g}:1 比例规则。"
        ),
        actual=instrument.accuracy_class,
        expected=f"<= {expected_max:g}",
    )


def validate_instrument(
    parameters: InstrumentParameters | dict[str, Any],
    instrument: InstrumentRecord | dict[str, Any],
    rules_path: str | Path | None = None,
) -> ValidationResult:
    params = (
        parameters
        if isinstance(parameters, InstrumentParameters)
        else InstrumentParameters.model_validate(parameters)
    )
    candidate = (
        instrument
        if isinstance(instrument, InstrumentRecord)
        else InstrumentRecord.model_validate(instrument)
    )
    rules = load_rules(str(rules_path or settings.rules_path))
    rule_type = canonicalize_instrument_type(params.instrument_type, rules)
    if not rule_type or rule_type not in rules:
        check = ValidationCheck(
            rule="rule_lookup",
            passed=False,
            reason=f"未找到被检仪器类型 {params.instrument_type} 对应的规则配置。",
        )
        return ValidationResult(passed=False, instrument_id=candidate.id, checks=[check])

    rule_spec = rules[rule_type]
    checks = [
        validate_type(params, candidate, rule_spec),
        validate_range(params, candidate, rule_spec),
        validate_accuracy(params, candidate, rule_spec),
    ]
    return ValidationResult(
        passed=all(check.passed for check in checks),
        instrument_id=candidate.id,
        checks=checks,
    )

