"""Small reusable constructors for common option-strategy examples."""
from __future__ import annotations

from black_scholes import OptionInputs
from option_portfolio import OptionLeg


def build_strategy(
    template: str,
    inputs: OptionInputs,
    width: float = 10.0,
) -> list[OptionLeg]:
    """Build common strategies from generic ``OptionLeg`` primitives.

    These are convenience templates for the interactive app. The underlying
    portfolio engine remains generic and can represent arbitrary collections of
    long/short call and put legs.
    """

    inputs.validate()
    if width <= 0:
        raise ValueError("width must be positive")

    base_kwargs = {
        "time_to_expiry": inputs.time_to_expiry,
        "volatility": inputs.volatility,
    }

    if template == "Long option":
        return [
            OptionLeg(
                strike=inputs.strike,
                option_type=inputs.option_type,
                quantity=1.0,
                **base_kwargs,
            )
        ]

    if template == "Vertical spread":
        second_strike = (
            inputs.strike + width
            if inputs.option_type == "call"
            else inputs.strike - width
        )
        if second_strike <= 0:
            raise ValueError("put-spread width makes the second strike non-positive")
        return [
            OptionLeg(
                strike=inputs.strike,
                option_type=inputs.option_type,
                quantity=1.0,
                **base_kwargs,
            ),
            OptionLeg(
                strike=second_strike,
                option_type=inputs.option_type,
                quantity=-1.0,
                **base_kwargs,
            ),
        ]

    if template == "Long straddle":
        return [
            OptionLeg(
                strike=inputs.strike,
                option_type="call",
                quantity=1.0,
                **base_kwargs,
            ),
            OptionLeg(
                strike=inputs.strike,
                option_type="put",
                quantity=1.0,
                **base_kwargs,
            ),
        ]

    raise ValueError(f"unknown strategy template: {template}")
