import pytest

from binomial import binomial_price
from black_scholes import OptionInputs, black_scholes_price


def inputs(option_type):
    return OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, option_type)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_tree_converges_to_black_scholes(option_type):
    model_inputs = inputs(option_type)
    tree = binomial_price(model_inputs, 1000, "european").price
    assert tree == pytest.approx(black_scholes_price(model_inputs), abs=0.01)


def test_american_put_not_below_european():
    model_inputs = inputs("put")
    european = binomial_price(model_inputs, 500, "european").price
    american = binomial_price(model_inputs, 500, "american").price
    assert american >= european


def test_non_dividend_american_call_matches_european():
    model_inputs = inputs("call")
    european = binomial_price(model_inputs, 800, "european").price
    american = binomial_price(model_inputs, 800, "american").price
    assert american == pytest.approx(european, abs=1e-8)
