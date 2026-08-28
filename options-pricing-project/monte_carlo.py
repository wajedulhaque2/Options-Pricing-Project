"""Risk-neutral Monte Carlo pricing for European options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from black_scholes import OptionInputs, intrinsic_value


@dataclass(frozen=True)
class MonteCarloResult:
    price: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    simulations: int
    confidence_level: float
    antithetic_variates: bool
    control_variate: bool


def monte_carlo_price(
    inputs: OptionInputs,
    simulations: int = 200_000,
    seed: int | None = 42,
    confidence_level: float = 0.95,
    antithetic_variates: bool = True,
    control_variate: bool = True,
) -> MonteCarloResult:
    """Estimate a European option value under risk-neutral GBM.

    When antithetic variates are enabled, each ``z`` and ``-z`` payoff pair is
    averaged before the sampling standard error is calculated. Treating the two
    members of an antithetic pair as independent would understate or otherwise
    distort the uncertainty estimate because they are intentionally correlated.
    """

    inputs.validate()
    if simulations < 2:
        raise ValueError("simulations must be at least two")
    if antithetic_variates and simulations % 2:
        raise ValueError(
            "antithetic variates require an even simulation count"
        )
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")

    if inputs.time_to_expiry == 0:
        value = intrinsic_value(inputs)
        return MonteCarloResult(
            value,
            0.0,
            value,
            value,
            simulations,
            confidence_level,
            antithetic_variates,
            control_variate,
        )

    rng = np.random.default_rng(seed)
    if antithetic_variates:
        half = simulations // 2
        base = rng.standard_normal(half)
        shocks = np.concatenate([base, -base])
    else:
        shocks = rng.standard_normal(simulations)

    drift = (
        inputs.risk_free_rate
        - inputs.dividend_yield
        - 0.5 * inputs.volatility**2
    ) * inputs.time_to_expiry
    diffusion = inputs.volatility * np.sqrt(inputs.time_to_expiry) * shocks
    terminal_spots = inputs.spot * np.exp(drift + diffusion)
    payoffs = (
        np.maximum(terminal_spots - inputs.strike, 0.0)
        if inputs.option_type == "call"
        else np.maximum(inputs.strike - terminal_spots, 0.0)
    )

    discount = np.exp(-inputs.risk_free_rate * inputs.time_to_expiry)
    discounted_payoffs = discount * payoffs
    adjusted = discounted_payoffs

    if control_variate:
        discounted_terminal = discount * terminal_spots
        known_expectation = inputs.spot * np.exp(
            -inputs.dividend_yield * inputs.time_to_expiry
        )
        variance_control = np.var(discounted_terminal, ddof=1)
        if variance_control > 0:
            covariance = np.cov(
                discounted_payoffs,
                discounted_terminal,
                ddof=1,
            )[0, 1]
            coefficient = covariance / variance_control
            adjusted = discounted_payoffs - coefficient * (
                discounted_terminal - known_expectation
            )

    if antithetic_variates:
        half = simulations // 2
        independent_estimates = (
            adjusted[:half] + adjusted[half:]
        ) / 2.0
        price = float(np.mean(independent_estimates))
        standard_error = float(
            np.std(independent_estimates, ddof=1) / np.sqrt(half)
        )
    else:
        price = float(np.mean(adjusted))
        standard_error = float(
            np.std(adjusted, ddof=1) / np.sqrt(simulations)
        )

    critical = float(norm.ppf(0.5 + confidence_level / 2.0))
    margin = critical * standard_error

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_low=float(price - margin),
        confidence_high=float(price + margin),
        simulations=simulations,
        confidence_level=confidence_level,
        antithetic_variates=antithetic_variates,
        control_variate=control_variate,
    )
