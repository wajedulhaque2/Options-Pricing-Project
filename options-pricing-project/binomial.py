"""Cox-Ross-Rubinstein recombining binomial-tree option pricing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from black_scholes import OptionInputs, intrinsic_value

ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True)
class BinomialResult:
    price: float
    steps: int
    up_factor: float
    down_factor: float
    risk_neutral_probability: float


def binomial_price(
    inputs: OptionInputs,
    steps: int = 500,
    exercise_style: ExerciseStyle = "european",
) -> BinomialResult:
    """Price a call or put with a CRR tree."""

    inputs.validate()
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if exercise_style not in {"european", "american"}:
        raise ValueError("exercise_style must be 'european' or 'american'")

    if inputs.time_to_expiry == 0:
        return BinomialResult(
            intrinsic_value(inputs), steps, 1.0, 1.0, 0.5
        )

    if inputs.volatility == 0:
        terminal_spot = inputs.spot * np.exp(
            (inputs.risk_free_rate - inputs.dividend_yield)
            * inputs.time_to_expiry
        )
        payoff = (
            max(terminal_spot - inputs.strike, 0.0)
            if inputs.option_type == "call"
            else max(inputs.strike - terminal_spot, 0.0)
        )
        european_value = np.exp(
            -inputs.risk_free_rate * inputs.time_to_expiry
        ) * payoff
        value = (
            max(european_value, intrinsic_value(inputs))
            if exercise_style == "american"
            else european_value
        )
        return BinomialResult(float(value), steps, 1.0, 1.0, 0.5)

    dt = inputs.time_to_expiry / steps
    up = np.exp(inputs.volatility * np.sqrt(dt))
    down = 1.0 / up
    growth = np.exp((inputs.risk_free_rate - inputs.dividend_yield) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "CRR probability is outside [0, 1]; increase steps or review inputs"
        )

    discount = np.exp(-inputs.risk_free_rate * dt)
    nodes = np.arange(steps + 1)
    terminal_spots = inputs.spot * up**nodes * down ** (steps - nodes)
    values = (
        np.maximum(terminal_spots - inputs.strike, 0.0)
        if inputs.option_type == "call"
        else np.maximum(inputs.strike - terminal_spots, 0.0)
    )

    for step in range(steps - 1, -1, -1):
        continuation = discount * (
            probability * values[1 : step + 2]
            + (1.0 - probability) * values[: step + 1]
        )
        if exercise_style == "european":
            values = continuation
            continue

        current_nodes = np.arange(step + 1)
        current_spots = inputs.spot * up**current_nodes * down ** (
            step - current_nodes
        )
        exercise = (
            np.maximum(current_spots - inputs.strike, 0.0)
            if inputs.option_type == "call"
            else np.maximum(inputs.strike - current_spots, 0.0)
        )
        values = np.maximum(continuation, exercise)

    return BinomialResult(
        price=float(values[0]),
        steps=steps,
        up_factor=float(up),
        down_factor=float(down),
        risk_neutral_probability=float(probability),
    )
