import pytest

from black_scholes import (
    OptionInputs,
    black_scholes_greeks,
    black_scholes_price,
    option_bounds,
    put_call_parity_residual,
)


def base(option_type="call"):
    return OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, option_type)


def test_known_call_value():
    assert black_scholes_price(base("call")) == pytest.approx(10.4506, abs=1e-4)


def test_known_put_value():
    assert black_scholes_price(base("put")) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    call = black_scholes_price(base("call"))
    put = black_scholes_price(base("put"))
    assert put_call_parity_residual(call, put, 100, 100, 1, 0.05) == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_within_bounds(option_type):
    inputs = base(option_type)
    lower, upper = option_bounds(inputs)
    assert lower <= black_scholes_price(inputs) <= upper


def test_call_delta_range():
    assert 0 < black_scholes_greeks(base("call"))["delta"] < 1


def test_put_delta_range():
    assert -1 < black_scholes_greeks(base("put"))["delta"] < 0
