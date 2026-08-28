import pytest

from black_scholes import OptionInputs
from strategy_templates import build_strategy


def inputs(option_type="call"):
    return OptionInputs(100, 100, 0.5, 0.04, 0.25, 0.01, option_type)


def test_call_vertical_uses_long_lower_and_short_higher_strike():
    legs = build_strategy("Vertical spread", inputs("call"), width=10)
    assert [(leg.strike, leg.quantity) for leg in legs] == [
        (100, 1.0),
        (110, -1.0),
    ]


def test_put_vertical_uses_long_higher_and_short_lower_strike():
    legs = build_strategy("Vertical spread", inputs("put"), width=10)
    assert [(leg.strike, leg.quantity) for leg in legs] == [
        (100, 1.0),
        (90, -1.0),
    ]


def test_long_straddle_has_call_and_put_at_same_strike():
    legs = build_strategy("Long straddle", inputs("call"), width=10)
    assert {leg.option_type for leg in legs} == {"call", "put"}
    assert {leg.strike for leg in legs} == {100}
    assert all(leg.quantity == 1 for leg in legs)


def test_vertical_rejects_nonpositive_second_put_strike():
    with pytest.raises(ValueError, match="non-positive"):
        build_strategy("Vertical spread", inputs("put"), width=150)
