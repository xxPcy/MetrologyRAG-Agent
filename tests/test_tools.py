from __future__ import annotations

from src.agent.agent import MetrologyAgentRunner
from src.models.schemas import ConversationState, InstrumentParameters
from src.tools.instrument_query_tool import query_instrument_catalog
from src.tools.parameter_extract_tool import extract_parameters
from src.tools.recommendation_tool import recommend_instruments
from src.tools.rule_validation_tool import validate_instrument_tool


def test_extract_parameters_from_pressure_question():
    params = extract_parameters("检定 0～1.6 MPa、1.6 级压力表应该选择什么标准器？")

    assert params["instrument_type"] == "pressure_gauge"
    assert params["range_min"] == 0
    assert params["range_max"] == 1.6
    assert params["unit"] == "MPa"
    assert params["accuracy_class"] == 1.6


def test_extract_parameters_does_not_take_range_from_context():
    params = extract_parameters(
        "检定压力表需要哪些标准器？",
        context="标准器最大允许误差不大于被检表最大允许误差的1/4，测量范围 0～60 MPa。",
    )

    assert params["instrument_type"] == "pressure_gauge"
    assert params["range_min"] is None
    assert params["range_max"] is None
    assert params["accuracy_class"] is None


def test_agent_merges_follow_up_parameters_from_state():
    runner = MetrologyAgentRunner()
    state = ConversationState(
        last_parameters=InstrumentParameters(
            instrument_type="pressure_gauge",
            range_min=0,
            range_max=1.6,
            unit="MPa",
            accuracy_class=1.6,
        )
    )
    current = InstrumentParameters(
        accuracy_class=2.5,
        accuracy_text="2.5 级",
        raw_text="那换成 2.5 级呢？",
    )

    merged = runner._merge_parameters_with_state("那换成 2.5 级呢？", current, state)

    assert merged.instrument_type == "pressure_gauge"
    assert merged.range_min == 0
    assert merged.range_max == 1.6
    assert merged.unit == "MPa"
    assert merged.accuracy_class == 2.5


def test_instrument_catalog_query_returns_demo_pressure_candidate():
    params = extract_parameters("检定 0～1.6 MPa、1.6 级压力表应该选择什么标准器？")
    candidates = query_instrument_catalog(params, limit=10)

    assert any(item["id"] == "D-PG-001" for item in candidates)


def test_rule_validation_tool_returns_passed_result():
    params = extract_parameters("检定 0～1.6 MPa、1.6 级压力表应该选择什么标准器？")
    instrument = {
        "id": "D-DP-001",
        "name": "数字压力计",
        "type": "digital_pressure_gauge",
        "model": "Demo-DP-001",
        "range_min": 0,
        "range_max": 2.5,
        "unit": "MPa",
        "accuracy_class": 0.05,
        "manufacturer": "Demo Manufacturer",
    }

    result = validate_instrument_tool(params, instrument)

    assert result["passed"] is True


def test_recommendation_tool_filters_failed_candidates():
    params = extract_parameters("检定 0～1.6 MPa、1.6 级压力表应该选择什么标准器？")
    candidates = [
        {
            "id": "OK",
            "name": "数字压力计",
            "type": "digital_pressure_gauge",
            "model": "Demo",
            "range_min": 0,
            "range_max": 2.5,
            "unit": "MPa",
            "accuracy_class": 0.05,
            "manufacturer": "Demo Manufacturer",
        },
        {
            "id": "BAD",
            "name": "精密压力表",
            "type": "precision_pressure_gauge",
            "model": "Demo",
            "range_min": 0,
            "range_max": 1.0,
            "unit": "MPa",
            "accuracy_class": 0.5,
            "manufacturer": "Demo Manufacturer",
        },
    ]

    recommendations = recommend_instruments(params, candidates)

    assert [item["instrument"]["id"] for item in recommendations] == ["OK"]
