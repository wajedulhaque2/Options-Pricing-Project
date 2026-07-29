from black_scholes import OptionInputs, black_scholes_price
from monte_carlo import monte_carlo_price


def test_confidence_interval_contains_analytical_value():
    inputs = OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, "call")
    analytical = black_scholes_price(inputs)
    result = monte_carlo_price(inputs, simulations=200_000, seed=123)
    assert result.confidence_low <= analytical <= result.confidence_high


def test_seed_is_reproducible():
    inputs = OptionInputs(100, 110, 0.5, 0.03, 0.25, 0.0, "put")
    first = monte_carlo_price(inputs, simulations=10_000, seed=9)
    second = monte_carlo_price(inputs, simulations=10_000, seed=9)
    assert first.price == second.price
    assert first.standard_error == second.standard_error
