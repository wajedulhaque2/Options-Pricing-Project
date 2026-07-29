"""Implied-volatility inversion using SciPy's bracketed Brent solver."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from black_scholes import OptionInputs, black_scholes_price, option_bounds


def implied_volatility(
    market_price: float,
    inputs: OptionInputs,
    lower_volatility: float = 1e-8,
    upper_volatility: float = 5.0,
    tolerance: float = 1e-10,
    maximum_iterations: int = 200,
) -> float:
    """Solve for volatility that matches a European option market price."""

    inputs.validate()
    if market_price < 0:
        raise ValueError("market_price cannot be negative")
    if inputs.time_to_expiry <= 0:
        raise ValueError("implied volatility requires positive time to expiry")
    if lower_volatility <= 0:
        raise ValueError("lower_volatility must be positive")
    if upper_volatility <= lower_volatility:
        raise ValueError("upper_volatility must exceed lower_volatility")

    lower_bound, upper_bound = option_bounds(inputs)
    epsilon = max(tolerance, 1e-12)

    if market_price < lower_bound - epsilon:
        raise ValueError(
            f"market_price is below the no-arbitrage lower bound {lower_bound:.8f}"
        )
    if market_price > upper_bound + epsilon:
        raise ValueError(
            f"market_price is above the no-arbitrage upper bound {upper_bound:.8f}"
        )
    if np.isclose(market_price, lower_bound, atol=epsilon, rtol=0.0):
        return float(lower_volatility)

    def pricing_error(volatility: float) -> float:
        trial = OptionInputs(
            spot=inputs.spot,
            strike=inputs.strike,
            time_to_expiry=inputs.time_to_expiry,
            risk_free_rate=inputs.risk_free_rate,
            volatility=volatility,
            dividend_yield=inputs.dividend_yield,
            option_type=inputs.option_type,
        )
        return black_scholes_price(trial) - market_price

    error_low = pricing_error(lower_volatility)
    error_high = pricing_error(upper_volatility)
    if error_low * error_high > 0:
        raise ValueError(
            "The selected volatility interval does not bracket a root"
        )

    return float(
        brentq(
            pricing_error,
            lower_volatility,
            upper_volatility,
            xtol=tolerance,
            maxiter=maximum_iterations,
        )
    )
