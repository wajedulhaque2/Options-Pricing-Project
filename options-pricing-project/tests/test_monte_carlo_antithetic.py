import numpy as np
import pytest

from black_scholes import OptionInputs
from monte_carlo import monte_carlo_price


def test_antithetic_requires_even_simulation_count():
    inputs = OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, "call")
    with pytest.raises(ValueError, match="even"):
        monte_carlo_price(
            inputs,
            simulations=999,
            antithetic_variates=True,
        )


def test_antithetic_standard_error_uses_pair_averages():
    inputs = OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, "call")
    simulations = 2_000
    seed = 123
    result = monte_carlo_price(
        inputs,
        simulations=simulations,
        seed=seed,
        antithetic_variates=True,
        control_variate=False,
    )

    rng = np.random.default_rng(seed)
    base = rng.standard_normal(simulations // 2)
    shocks = np.concatenate([base, -base])
    terminal = 100 * np.exp((0.05 - 0.5 * 0.20**2) + 0.20 * shocks)
    discounted = np.exp(-0.05) * np.maximum(terminal - 100, 0)
    pair_means = (
        discounted[: simulations // 2] + discounted[simulations // 2 :]
    ) / 2
    expected = pair_means.std(ddof=1) / np.sqrt(simulations // 2)

    assert result.standard_error == pytest.approx(expected, rel=1e-12)
