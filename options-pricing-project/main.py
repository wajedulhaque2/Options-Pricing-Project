"""Complete options-pricing research workflow.

Edit SETTINGS, then run:
    python main.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from analytics import (
    american_early_exercise_premium,
    greek_profile,
    price_surface,
    pricing_comparison,
)
from black_scholes import (
    OptionInputs,
    black_scholes_greeks,
    black_scholes_price,
    option_bounds,
    put_call_parity_residual,
)
from implied_volatility import implied_volatility


# ==================================================
# SETTINGS
# ==================================================

OUTPUT_DIRECTORY = Path("outputs")
SHOW_PLOTS = True

INPUTS = OptionInputs(
    spot=100.0,
    strike=100.0,
    time_to_expiry=0.5,
    risk_free_rate=0.05,
    volatility=0.20,
    dividend_yield=0.01,
    option_type="call",
)

BINOMIAL_STEPS = 600
MONTE_CARLO_SIMULATIONS = 250_000
MONTE_CARLO_SEED = 42


def save_surface_chart(surface, output_path: Path) -> None:
    spots = surface.columns.astype(float).to_numpy()
    volatilities = surface.index.to_numpy(dtype=float)
    x_values, y_values = np.meshgrid(spots, volatilities)

    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(x_values, y_values, surface.to_numpy(), cmap="viridis")
    axis.set_title("Black-Scholes option-price surface")
    axis.set_xlabel("Spot price")
    axis.set_ylabel("Volatility")
    axis.set_zlabel("Option price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_greek_chart(profile, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    columns = [
        ("price", "Option price"),
        ("delta", "Delta"),
        ("gamma", "Gamma"),
        ("vega_per_1pct", "Vega per 1 vol point"),
        ("theta_per_day", "Theta per day"),
        ("rho_per_1pct", "Rho per 1 rate point"),
    ]
    for axis, (column, title) in zip(axes.flat, columns):
        axis.plot(profile["spot"], profile[column])
        axis.set_title(title)
        axis.set_xlabel("Spot")
        axis.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def main() -> dict[str, Any]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    INPUTS.validate()

    call_inputs = OptionInputs(**{**asdict(INPUTS), "option_type": "call"})
    put_inputs = OptionInputs(**{**asdict(INPUTS), "option_type": "put"})

    call_price = black_scholes_price(call_inputs)
    put_price = black_scholes_price(put_inputs)
    call_greeks = black_scholes_greeks(call_inputs)
    put_greeks = black_scholes_greeks(put_inputs)
    call_bounds = option_bounds(call_inputs)
    put_bounds = option_bounds(put_inputs)

    parity_residual = put_call_parity_residual(
        call_price=call_price,
        put_price=put_price,
        spot=INPUTS.spot,
        strike=INPUTS.strike,
        time_to_expiry=INPUTS.time_to_expiry,
        risk_free_rate=INPUTS.risk_free_rate,
        dividend_yield=INPUTS.dividend_yield,
    )

    recovered_iv = implied_volatility(
        market_price=black_scholes_price(INPUTS),
        inputs=INPUTS,
    )

    comparison = pricing_comparison(
        INPUTS,
        binomial_steps=BINOMIAL_STEPS,
        monte_carlo_simulations=MONTE_CARLO_SIMULATIONS,
        monte_carlo_seed=MONTE_CARLO_SEED,
    )
    early_exercise = american_early_exercise_premium(
        put_inputs,
        steps=BINOMIAL_STEPS,
    )

    spot_values = np.linspace(INPUTS.spot * 0.60, INPUTS.spot * 1.40, 41)
    volatility_values = np.linspace(0.05, 0.80, 31)
    surface = price_surface(INPUTS, spot_values, volatility_values)
    profile = greek_profile(INPUTS, spot_values)

    summary = {
        "inputs": asdict(INPUTS),
        "call_price": call_price,
        "put_price": put_price,
        "call_bounds": call_bounds,
        "put_bounds": put_bounds,
        "put_call_parity_residual": parity_residual,
        "recovered_implied_volatility": recovered_iv,
        "call_greeks": call_greeks,
        "put_greeks": put_greeks,
        "american_put_comparison": early_exercise,
    }

    comparison.to_csv(OUTPUT_DIRECTORY /
                      "model_price_comparison.csv", index=False)
    surface.to_csv(OUTPUT_DIRECTORY / "option_price_surface.csv")
    profile.to_csv(OUTPUT_DIRECTORY / "greek_profile.csv", index=False)
    with (OUTPUT_DIRECTORY / "calculation_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    save_surface_chart(surface, OUTPUT_DIRECTORY / "option_price_surface.png")
    save_greek_chart(profile, OUTPUT_DIRECTORY / "greek_profile.png")

    print("\n" + "=" * 72)
    print("OPTIONS PRICING CALCULATOR")
    print("=" * 72)
    print(json.dumps(asdict(INPUTS), indent=2))
    print("\nMODEL PRICE COMPARISON")
    print(comparison.round(6).to_string(index=False))
    print("\nBLACK-SCHOLES PRICES")
    print(f"Call: {call_price:.6f}")
    print(f"Put:  {put_price:.6f}")
    print("\nPUT-CALL PARITY")
    print(f"Residual: {parity_residual:.12f}")
    print("\nRECOVERED IMPLIED VOLATILITY")
    print(f"{recovered_iv:.6%}")
    print("\nCALL GREEKS")
    for name, value in call_greeks.items():
        print(f"{name}: {value:.6f}")
    print("\nAMERICAN PUT")
    for name, value in early_exercise.items():
        print(f"{name}: {value:.6f}")
    print(f"\nFiles saved to: {OUTPUT_DIRECTORY.resolve()}")

    return {
        "summary": summary,
        "comparison": comparison,
        "surface": surface,
        "greek_profile": profile,
    }


if __name__ == "__main__":
    main()
