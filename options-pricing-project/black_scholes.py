"""Black-Scholes-Merton pricing, Greeks, bounds, and parity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionInputs:
    """Inputs shared by all option-pricing models."""

    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0
    option_type: OptionType = "call"

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be greater than zero")
        if self.strike <= 0:
            raise ValueError("strike must be greater than zero")
        if self.time_to_expiry < 0:
            raise ValueError("time_to_expiry cannot be negative")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.risk_free_rate <= -1:
            raise ValueError("risk_free_rate must be greater than -100%")
        if self.dividend_yield <= -1:
            raise ValueError("dividend_yield must be greater than -100%")
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")


def intrinsic_value(inputs: OptionInputs) -> float:
    """Return the immediate-exercise payoff."""

    inputs.validate()
    if inputs.option_type == "call":
        return max(inputs.spot - inputs.strike, 0.0)
    return max(inputs.strike - inputs.spot, 0.0)


def d1_d2(inputs: OptionInputs) -> tuple[float, float]:
    """Return the Black-Scholes-Merton d1 and d2 terms."""

    inputs.validate()
    if inputs.time_to_expiry <= 0:
        raise ValueError("d1 and d2 are undefined at expiry")
    if inputs.volatility <= 0:
        raise ValueError("d1 and d2 require positive volatility")

    sqrt_time = np.sqrt(inputs.time_to_expiry)
    d1 = (
        np.log(inputs.spot / inputs.strike)
        + (
            inputs.risk_free_rate
            - inputs.dividend_yield
            + 0.5 * inputs.volatility**2
        )
        * inputs.time_to_expiry
    ) / (inputs.volatility * sqrt_time)
    d2 = d1 - inputs.volatility * sqrt_time
    return float(d1), float(d2)


def option_bounds(inputs: OptionInputs) -> tuple[float, float]:
    """Return European no-arbitrage lower and upper price bounds."""

    inputs.validate()
    if inputs.time_to_expiry == 0:
        value = intrinsic_value(inputs)
        return value, value

    discounted_spot = inputs.spot * np.exp(
        -inputs.dividend_yield * inputs.time_to_expiry
    )
    discounted_strike = inputs.strike * np.exp(
        -inputs.risk_free_rate * inputs.time_to_expiry
    )

    if inputs.option_type == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    else:
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike
    return float(lower), float(upper)


def black_scholes_price(inputs: OptionInputs) -> float:
    """Price a European call or put with continuous dividend yield."""

    inputs.validate()
    if inputs.time_to_expiry == 0:
        return intrinsic_value(inputs)

    discounted_spot = inputs.spot * np.exp(
        -inputs.dividend_yield * inputs.time_to_expiry
    )
    discounted_strike = inputs.strike * np.exp(
        -inputs.risk_free_rate * inputs.time_to_expiry
    )

    if inputs.volatility == 0:
        if inputs.option_type == "call":
            return float(max(discounted_spot - discounted_strike, 0.0))
        return float(max(discounted_strike - discounted_spot, 0.0))

    d1, d2 = d1_d2(inputs)
    if inputs.option_type == "call":
        price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    else:
        price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    return float(price)


def black_scholes_greeks(inputs: OptionInputs) -> dict[str, float]:
    """Return analytical Greeks.

    Vega and rho are reported per one percentage-point change. Theta is
    returned per year and per calendar day.
    """

    inputs.validate()
    if inputs.time_to_expiry <= 0:
        raise ValueError("Greeks are not defined by this function at expiry")
    if inputs.volatility <= 0:
        raise ValueError("Greeks require positive volatility")

    d1, d2 = d1_d2(inputs)
    sqrt_time = np.sqrt(inputs.time_to_expiry)
    discount_spot = np.exp(-inputs.dividend_yield * inputs.time_to_expiry)
    discount_strike = np.exp(-inputs.risk_free_rate * inputs.time_to_expiry)
    density = norm.pdf(d1)

    gamma = discount_spot * density / (
        inputs.spot * inputs.volatility * sqrt_time
    )
    vega_raw = inputs.spot * discount_spot * density * sqrt_time
    common_theta = -(
        inputs.spot
        * discount_spot
        * density
        * inputs.volatility
        / (2.0 * sqrt_time)
    )

    if inputs.option_type == "call":
        delta = discount_spot * norm.cdf(d1)
        theta_annual = (
            common_theta
            - inputs.risk_free_rate
            * inputs.strike
            * discount_strike
            * norm.cdf(d2)
            + inputs.dividend_yield
            * inputs.spot
            * discount_spot
            * norm.cdf(d1)
        )
        rho_raw = (
            inputs.strike
            * inputs.time_to_expiry
            * discount_strike
            * norm.cdf(d2)
        )
    else:
        delta = discount_spot * (norm.cdf(d1) - 1.0)
        theta_annual = (
            common_theta
            + inputs.risk_free_rate
            * inputs.strike
            * discount_strike
            * norm.cdf(-d2)
            - inputs.dividend_yield
            * inputs.spot
            * discount_spot
            * norm.cdf(-d1)
        )
        rho_raw = -(
            inputs.strike
            * inputs.time_to_expiry
            * discount_strike
            * norm.cdf(-d2)
        )

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega_per_1pct": float(vega_raw / 100.0),
        "theta_per_year": float(theta_annual),
        "theta_per_day": float(theta_annual / 365.0),
        "rho_per_1pct": float(rho_raw / 100.0),
    }


def put_call_parity_residual(
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
) -> float:
    """Return the difference between the two sides of put-call parity."""

    left = call_price - put_price
    right = (
        spot * np.exp(-dividend_yield * time_to_expiry)
        - strike * np.exp(-risk_free_rate * time_to_expiry)
    )
    return float(left - right)
