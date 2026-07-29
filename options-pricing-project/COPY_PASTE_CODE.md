# Copy-and-paste code for every project file

## `black_scholes.py`

```python
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
```

## `implied_volatility.py`

```python
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
```

## `binomial.py`

```python
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
```

## `monte_carlo.py`

```python
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
    """Estimate a European option value under risk-neutral GBM."""

    inputs.validate()
    if simulations < 2:
        raise ValueError("simulations must be at least two")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one")

    if inputs.time_to_expiry == 0:
        value = intrinsic_value(inputs)
        return MonteCarloResult(
            value, 0.0, value, value, simulations,
            confidence_level, antithetic_variates, control_variate
        )

    rng = np.random.default_rng(seed)
    if antithetic_variates:
        half = (simulations + 1) // 2
        base = rng.standard_normal(half)
        shocks = np.concatenate([base, -base])[:simulations]
    else:
        shocks = rng.standard_normal(simulations)

    drift = (
        inputs.risk_free_rate
        - inputs.dividend_yield
        - 0.5 * inputs.volatility**2
    ) * inputs.time_to_expiry
    diffusion = (
        inputs.volatility * np.sqrt(inputs.time_to_expiry) * shocks
    )
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
                discounted_payoffs, discounted_terminal, ddof=1
            )[0, 1]
            coefficient = covariance / variance_control
            adjusted = discounted_payoffs - coefficient * (
                discounted_terminal - known_expectation
            )

    price = float(np.mean(adjusted))
    standard_error = float(np.std(adjusted, ddof=1) / np.sqrt(simulations))
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
```

## `analytics.py`

```python
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
```

## `market_data.py`

```python
"""Optional Yahoo Finance spot, history, and option-chain helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class OptionChainData:
    ticker: str
    expiration: str
    spot: float
    calls: pd.DataFrame
    puts: pd.DataFrame


def download_adjusted_history(ticker: str, period: str = "1y") -> pd.Series:
    data = yf.Ticker(ticker).history(
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
    )
    if data.empty or "Close" not in data:
        raise ValueError(f"No historical prices returned for {ticker}")
    prices = pd.to_numeric(data["Close"], errors="coerce").dropna().astype(float)
    prices.name = ticker.upper()
    return prices


def latest_spot(ticker: str) -> float:
    return float(download_adjusted_history(ticker, period="1mo").iloc[-1])


def historical_volatility(
    ticker: str,
    period: str = "1y",
    trading_days_per_year: int = 252,
) -> float:
    prices = download_adjusted_history(ticker, period=period)
    log_returns = np.log(prices).diff().dropna()
    if len(log_returns) < 2:
        raise ValueError("Not enough observations to estimate volatility")
    return float(log_returns.std(ddof=1) * np.sqrt(trading_days_per_year))


def option_expirations(ticker: str) -> tuple[str, ...]:
    expirations = tuple(yf.Ticker(ticker).options)
    if not expirations:
        raise ValueError(f"No option expirations returned for {ticker}")
    return expirations


def download_option_chain(ticker: str, expiration: str) -> OptionChainData:
    security = yf.Ticker(ticker)
    available = tuple(security.options)
    if expiration not in available:
        raise ValueError(
            f"Expiration {expiration} is unavailable. "
            f"First available values: {available[:5]}"
        )

    chain = security.option_chain(expiration)
    calls = chain.calls.copy()
    puts = chain.puts.copy()

    for frame in (calls, puts):
        if {"bid", "ask"}.issubset(frame.columns):
            valid = (frame["bid"] > 0) & (frame["ask"] > 0)
            frame["midpoint"] = np.where(
                valid,
                (frame["bid"] + frame["ask"]) / 2.0,
                frame.get("lastPrice", np.nan),
            )

    return OptionChainData(
        ticker=ticker.upper(),
        expiration=expiration,
        spot=latest_spot(ticker),
        calls=calls,
        puts=puts,
    )
```

## `main.py`

```python
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

    comparison.to_csv(OUTPUT_DIRECTORY / "model_price_comparison.csv", index=False)
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
```

## `app.py`

```python
"""Interactive Streamlit options-pricing calculator."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

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
)
from implied_volatility import implied_volatility
from market_data import (
    download_option_chain,
    historical_volatility,
    option_expirations,
)

st.set_page_config(page_title="Options Pricing Calculator", layout="wide")
st.title("Options Pricing Calculator")
st.caption(
    "Black-Scholes-Merton, CRR trees, Monte Carlo, Greeks, implied volatility, "
    "scenario analysis, and optional Yahoo option chains."
)

with st.sidebar:
    st.header("Contract inputs")
    option_type = st.selectbox("Option type", ["call", "put"])
    spot = st.number_input("Spot price", min_value=0.01, value=100.0, step=1.0)
    strike = st.number_input("Strike price", min_value=0.01, value=100.0, step=1.0)
    time_to_expiry = st.number_input(
        "Time to expiry in years",
        min_value=0.0001,
        value=0.5,
        step=0.05,
        format="%.4f",
    )
    risk_free_rate = st.number_input(
        "Risk-free rate", value=0.05, step=0.005, format="%.4f"
    )
    dividend_yield = st.number_input(
        "Continuous dividend yield", value=0.01, step=0.005, format="%.4f"
    )
    volatility = st.number_input(
        "Volatility", min_value=0.0001, value=0.20, step=0.01, format="%.4f"
    )
    st.header("Numerical settings")
    binomial_steps = st.slider("Binomial steps", 50, 2_000, 500, 50)
    simulations = st.slider("Monte Carlo simulations", 10_000, 500_000, 100_000, 10_000)

inputs = OptionInputs(
    spot=float(spot),
    strike=float(strike),
    time_to_expiry=float(time_to_expiry),
    risk_free_rate=float(risk_free_rate),
    volatility=float(volatility),
    dividend_yield=float(dividend_yield),
    option_type=option_type,
)

calculator_tab, greeks_tab, iv_tab, surface_tab, american_tab, market_tab = st.tabs(
    [
        "Calculator",
        "Greeks",
        "Implied volatility",
        "Scenario surface",
        "American exercise",
        "Live option chain",
    ]
)

with calculator_tab:
    comparison = pricing_comparison(inputs, binomial_steps, simulations)
    lower, upper = option_bounds(inputs)
    cols = st.columns(3)
    cols[0].metric("Black-Scholes price", f"{black_scholes_price(inputs):,.4f}")
    cols[1].metric("Lower bound", f"{lower:,.4f}")
    cols[2].metric("Upper bound", f"{upper:,.4f}")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

with greeks_tab:
    st.dataframe(
        pd.Series(black_scholes_greeks(inputs), name="value").to_frame(),
        use_container_width=True,
    )
    spots = np.linspace(inputs.spot * 0.5, inputs.spot * 1.5, 101)
    profile = greek_profile(inputs, spots)
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    plots = [
        ("price", "Option price"),
        ("delta", "Delta"),
        ("gamma", "Gamma"),
        ("vega_per_1pct", "Vega per 1 vol point"),
        ("theta_per_day", "Theta per day"),
        ("rho_per_1pct", "Rho per 1 rate point"),
    ]
    for axis, (column, title) in zip(axes.flat, plots):
        axis.plot(profile["spot"], profile[column])
        axis.axvline(inputs.strike, linestyle="--")
        axis.set_title(title)
        axis.grid(True)
    plt.tight_layout()
    st.pyplot(figure)

with iv_tab:
    market_price = st.number_input(
        "Observed market price",
        min_value=0.0,
        value=float(black_scholes_price(inputs)),
        step=0.10,
    )
    try:
        result = implied_volatility(float(market_price), inputs)
        st.metric("Implied volatility", f"{result:.4%}")
    except ValueError as error:
        st.error(str(error))

with surface_tab:
    spots = np.linspace(inputs.spot * 0.60, inputs.spot * 1.40, 41)
    vols = np.linspace(0.05, 0.80, 31)
    surface = price_surface(inputs, spots, vols)
    figure, axis = plt.subplots(figsize=(11, 6))
    image = axis.imshow(
        surface.to_numpy(),
        aspect="auto",
        origin="lower",
        extent=[spots.min(), spots.max(), vols.min(), vols.max()],
    )
    axis.set_xlabel("Spot")
    axis.set_ylabel("Volatility")
    axis.set_title("Black-Scholes option-price surface")
    figure.colorbar(image, ax=axis, label="Option price")
    st.pyplot(figure)

with american_tab:
    st.dataframe(
        pd.Series(
            american_early_exercise_premium(inputs, binomial_steps),
            name="value",
        ).to_frame(),
        use_container_width=True,
    )

with market_tab:
    ticker = st.text_input("Ticker", value="AAPL").upper()
    if st.button("Load expirations"):
        try:
            st.session_state["expirations"] = option_expirations(ticker)
        except Exception as error:
            st.error(str(error))
    expirations = st.session_state.get("expirations", ())
    if expirations:
        expiration = st.selectbox("Expiration", expirations)
        if st.button("Load option chain"):
            try:
                chain = download_option_chain(ticker, expiration)
                st.metric(f"{ticker} latest adjusted close", f"{chain.spot:,.2f}")
                calls_tab, puts_tab = st.tabs(["Calls", "Puts"])
                with calls_tab:
                    st.dataframe(chain.calls, use_container_width=True)
                with puts_tab:
                    st.dataframe(chain.puts, use_container_width=True)
                try:
                    st.metric(
                        "One-year historical volatility",
                        f"{historical_volatility(ticker):.2%}",
                    )
                except Exception:
                    pass
            except Exception as error:
                st.error(str(error))
```

## `requirements.txt`

```text
numpy>=2.0
pandas>=2.2
scipy>=1.13
matplotlib>=3.9
yfinance>=0.2
streamlit>=1.40
jupyter>=1.1
pytest>=8.0
```

## `pytest.ini`

```text
[pytest]
pythonpath = .
```

## `tests/test_black_scholes.py`

```python
import pytest

from black_scholes import (
    OptionInputs,
    black_scholes_greeks,
    black_scholes_price,
    option_bounds,
    put_call_parity_residual,
)


def base(option_type="call"):
    return OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, option_type)


def test_known_call_value():
    assert black_scholes_price(base("call")) == pytest.approx(10.4506, abs=1e-4)


def test_known_put_value():
    assert black_scholes_price(base("put")) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    call = black_scholes_price(base("call"))
    put = black_scholes_price(base("put"))
    assert put_call_parity_residual(call, put, 100, 100, 1, 0.05) == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_price_within_bounds(option_type):
    inputs = base(option_type)
    lower, upper = option_bounds(inputs)
    assert lower <= black_scholes_price(inputs) <= upper


def test_call_delta_range():
    assert 0 < black_scholes_greeks(base("call"))["delta"] < 1


def test_put_delta_range():
    assert -1 < black_scholes_greeks(base("put"))["delta"] < 0
```

## `tests/test_binomial.py`

```python
import pytest

from binomial import binomial_price
from black_scholes import OptionInputs, black_scholes_price


def inputs(option_type):
    return OptionInputs(100, 100, 1, 0.05, 0.20, 0.0, option_type)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_tree_converges_to_black_scholes(option_type):
    model_inputs = inputs(option_type)
    tree = binomial_price(model_inputs, 1000, "european").price
    assert tree == pytest.approx(black_scholes_price(model_inputs), abs=0.01)


def test_american_put_not_below_european():
    model_inputs = inputs("put")
    european = binomial_price(model_inputs, 500, "european").price
    american = binomial_price(model_inputs, 500, "american").price
    assert american >= european


def test_non_dividend_american_call_matches_european():
    model_inputs = inputs("call")
    european = binomial_price(model_inputs, 800, "european").price
    american = binomial_price(model_inputs, 800, "american").price
    assert american == pytest.approx(european, abs=1e-8)
```

## `tests/test_implied_volatility.py`

```python
import pytest

from black_scholes import OptionInputs, black_scholes_price
from implied_volatility import implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_original_volatility(option_type):
    inputs = OptionInputs(125, 120, 0.75, 0.04, 0.32, 0.015, option_type)
    price = black_scholes_price(inputs)
    assert implied_volatility(price, inputs) == pytest.approx(0.32, abs=1e-8)


def test_rejects_price_above_upper_bound():
    inputs = OptionInputs(100, 100, 1, 0.05, 0.2, 0.0, "call")
    with pytest.raises(ValueError):
        implied_volatility(200, inputs)
```

## `tests/test_monte_carlo.py`

```python
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
```

## `tests/test_validation.py`

```python
import pytest

from black_scholes import OptionInputs, black_scholes_price


@pytest.mark.parametrize(
    "field,value",
    [("spot", 0), ("strike", 0), ("time_to_expiry", -1), ("volatility", -0.1)],
)
def test_invalid_inputs_rejected(field, value):
    values = {
        "spot": 100,
        "strike": 100,
        "time_to_expiry": 1,
        "risk_free_rate": 0.05,
        "volatility": 0.2,
        "option_type": "call",
    }
    values[field] = value
    with pytest.raises(ValueError):
        black_scholes_price(OptionInputs(**values))
```
