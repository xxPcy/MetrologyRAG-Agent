from __future__ import annotations

from src.rules.validator import validate_instrument


BASE_PARAMS = {
    "instrument_type": "pressure_gauge",
    "range_min": 0,
    "range_max": 1.6,
    "unit": "MPa",
    "accuracy_class": 1.6,
}


def _instrument(**overrides):
    data = {
        "id": "T-001",
        "name": "数字压力计",
        "type": "digital_pressure_gauge",
        "model": "Demo",
        "range_min": 0,
        "range_max": 2.5,
        "unit": "MPa",
        "accuracy_class": 0.05,
        "manufacturer": "Demo Manufacturer",
    }
    data.update(overrides)
    return data


def test_range_passes_when_standard_covers_dut_range():
    result = validate_instrument(BASE_PARAMS, _instrument(range_min=-0.1, range_max=2.5))

    assert result.passed is True
    assert next(check for check in result.checks if check.rule == "range").passed is True


def test_range_fails_when_standard_does_not_cover_dut_range():
    result = validate_instrument(BASE_PARAMS, _instrument(range_min=0, range_max=1.0))

    assert result.passed is False
    assert next(check for check in result.checks if check.rule == "range").passed is False


def test_accuracy_passes_when_ratio_is_satisfied():
    result = validate_instrument(BASE_PARAMS, _instrument(accuracy_class=0.25))

    assert result.passed is True
    assert next(check for check in result.checks if check.rule == "accuracy").passed is True


def test_accuracy_fails_when_ratio_is_not_satisfied():
    result = validate_instrument(BASE_PARAMS, _instrument(accuracy_class=0.5))

    assert result.passed is False
    assert next(check for check in result.checks if check.rule == "accuracy").passed is False

