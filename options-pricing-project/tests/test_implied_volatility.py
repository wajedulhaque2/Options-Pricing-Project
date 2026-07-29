import pytest

from black_scholes import OptionInputs, black_scholes_price
from implied_volatility import implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_original_volatility(option_type):
    inputs = OptionInputs(125, 120, 0.75, 0.04, 0.32, 0.015, option_type)
    price = black_scholes_price(inputs)
    assert implied_volatility(price, inputs) == pytest.approx(0.32, abs=1e-8)


def test_rejects_price_above_upper_bound():
    inputs = OptionInputs(100, 100, 1, 0.05, 0.2, 0.0, "call")
    with pytest.raises(ValueError):
        implied_volatility(200, inputs)
