"""Generic multi-leg option portfolio valuation, Greeks and scenario P&L."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from black_scholes import OptionInputs, black_scholes_greeks, black_scholes_price


@dataclass(frozen=True)
class OptionLeg:
    strike: float
    time_to_expiry: float
    volatility: float
    option_type: str
    quantity: float = 1.0
    contract_multiplier: float = 100.0

    def validate(self) -> None:
        if self.strike <= 0 or self.time_to_expiry < 0 or self.volatility < 0:
            raise ValueError(
                "strike must be positive; time and volatility non-negative"
            )
        if self.option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive")


def _inputs(
    leg: OptionLeg,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> OptionInputs:
    leg.validate()
    return OptionInputs(
        spot=spot,
        strike=leg.strike,
        time_to_expiry=leg.time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=leg.volatility,
        dividend_yield=dividend_yield,
        option_type=leg.option_type,
    )


def portfolio_value(
    legs: Iterable[OptionLeg],
    spot: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    total = 0.0
    for leg in legs:
        value = black_scholes_price(
            _inputs(leg, spot, risk_free_rate, dividend_yield)
        )
        total += leg.quantity * leg.contract_multiplier * value
    return float(total)


def portfolio_greeks(
    legs: Iterable[OptionLeg],
    spot: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> dict[str, float]:
    totals = {
        "delta": 0.0,
        "gamma": 0.0,
        "vega_per_1pct": 0.0,
        "theta_per_day": 0.0,
        "rho_per_1pct": 0.0,
    }
    for leg in legs:
        greeks = black_scholes_greeks(
            _inputs(leg, spot, risk_free_rate, dividend_yield)
        )
        scale = leg.quantity * leg.contract_multiplier
        for key in totals:
            totals[key] += scale * greeks[key]
    return {key: float(value) for key, value in totals.items()}


def expiration_value(
    legs: Iterable[OptionLeg],
    spot_values: np.ndarray,
) -> pd.Series:
    spots = np.asarray(spot_values, dtype=float)
    if (spots < 0).any():
        raise ValueError("spot_values cannot be negative")
    total = np.zeros_like(spots, dtype=float)
    for leg in legs:
        leg.validate()
        payoff = (
            np.maximum(spots - leg.strike, 0.0)
            if leg.option_type == "call"
            else np.maximum(leg.strike - spots, 0.0)
        )
        total += leg.quantity * leg.contract_multiplier * payoff
    return pd.Series(total, index=spots, name="expiration_value")


def scenario_grid(
    legs: Iterable[OptionLeg],
    base_spot: float,
    spot_shocks: Iterable[float],
    volatility_shocks: Iterable[float],
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    days_forward: int = 0,
) -> pd.DataFrame:
    """Reprice the portfolio across spot and absolute volatility shocks."""
    legs = list(legs)
    if not legs:
        raise ValueError("legs cannot be empty")
    if base_spot <= 0 or days_forward < 0:
        raise ValueError(
            "base_spot must be positive and days_forward non-negative"
        )

    base_value = portfolio_value(
        legs, base_spot, risk_free_rate, dividend_yield
    )
    rows = []
    for spot_shock in spot_shocks:
        shocked_spot = base_spot * (1.0 + float(spot_shock))
        if shocked_spot <= 0:
            continue
        for vol_shock in volatility_shocks:
            shocked_legs = []
            for leg in legs:
                shocked_vol = max(leg.volatility + float(vol_shock), 1e-8)
                shocked_time = max(
                    leg.time_to_expiry - days_forward / 365.0, 0.0
                )
                shocked_legs.append(
                    replace(
                        leg,
                        volatility=shocked_vol,
                        time_to_expiry=shocked_time,
                    )
                )
            value = portfolio_value(
                shocked_legs,
                shocked_spot,
                risk_free_rate,
                dividend_yield,
            )
            rows.append(
                {
                    "spot_shock": float(spot_shock),
                    "volatility_shock": float(vol_shock),
                    "days_forward": int(days_forward),
                    "spot": float(shocked_spot),
                    "portfolio_value": float(value),
                    "pnl": float(value - base_value),
                }
            )
    return pd.DataFrame(rows)
