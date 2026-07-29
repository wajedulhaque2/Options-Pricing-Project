"""Scenario grids, Greek profiles, and model-comparison helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from black_scholes import OptionInputs, black_scholes_greeks, black_scholes_price
from binomial import binomial_price
from monte_carlo import monte_carlo_price


def pricing_comparison(
    inputs: OptionInputs,
    binomial_steps: int = 500,
    monte_carlo_simulations: int = 200_000,
    monte_carlo_seed: int = 42,
) -> pd.DataFrame:
    analytical = black_scholes_price(inputs)
    tree = binomial_price(
        inputs, steps=binomial_steps, exercise_style="european"
    ).price
    simulation = monte_carlo_price(
        inputs,
        simulations=monte_carlo_simulations,
        seed=monte_carlo_seed,
        antithetic_variates=True,
        control_variate=True,
    )
    return pd.DataFrame(
        [
            {
                "method": "Black-Scholes-Merton",
                "price": analytical,
                "standard_error": np.nan,
                "confidence_low": np.nan,
                "confidence_high": np.nan,
            },
            {
                "method": "CRR Binomial — European",
                "price": tree,
                "standard_error": np.nan,
                "confidence_low": np.nan,
                "confidence_high": np.nan,
            },
            {
                "method": "Monte Carlo",
                "price": simulation.price,
                "standard_error": simulation.standard_error,
                "confidence_low": simulation.confidence_low,
                "confidence_high": simulation.confidence_high,
            },
        ]
    )


def price_surface(
    inputs: OptionInputs,
    spot_values: np.ndarray,
    volatility_values: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for volatility in volatility_values:
        row: dict[str, float] = {"volatility": float(volatility)}
        for spot in spot_values:
            trial = replace(inputs, spot=float(spot), volatility=float(volatility))
            row[f"{spot:.4f}"] = black_scholes_price(trial)
        records.append(row)
    return pd.DataFrame(records).set_index("volatility")


def greek_profile(
    inputs: OptionInputs,
    spot_values: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for spot in spot_values:
        trial = replace(inputs, spot=float(spot))
        rows.append(
            {
                "spot": float(spot),
                "price": black_scholes_price(trial),
                **black_scholes_greeks(trial),
            }
        )
    return pd.DataFrame(rows)


def american_early_exercise_premium(
    inputs: OptionInputs,
    steps: int = 800,
) -> dict[str, float]:
    european = binomial_price(
        inputs, steps=steps, exercise_style="european"
    ).price
    american = binomial_price(
        inputs, steps=steps, exercise_style="american"
    ).price
    return {
        "european_price": european,
        "american_price": american,
        "early_exercise_premium": american - european,
    }
