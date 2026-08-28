"""Option-chain cleaning and implied-volatility/Greek analytics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from black_scholes import OptionInputs, black_scholes_greeks, option_bounds
from implied_volatility import implied_volatility


@dataclass(frozen=True)
class ChainFilterConfig:
    max_relative_spread: float = 0.35
    minimum_open_interest: int = 0
    minimum_volume: int = 0

    def validate(self) -> None:
        if self.max_relative_spread <= 0:
            raise ValueError("max_relative_spread must be positive")
        if self.minimum_open_interest < 0 or self.minimum_volume < 0:
            raise ValueError("liquidity thresholds cannot be negative")


def clean_option_quotes(
    quotes: pd.DataFrame,
    config: ChainFilterConfig | None = None,
) -> pd.DataFrame:
    """Remove stale/illiquid quotes and calculate a defensible midpoint."""
    config = config or ChainFilterConfig()
    config.validate()
    required = {"strike", "bid", "ask"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"quotes is missing required columns: {sorted(missing)}")

    clean = quotes.copy()
    for column in [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
    ]:
        if column in clean:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean[
        (clean["strike"] > 0)
        & (clean["bid"] > 0)
        & (clean["ask"] >= clean["bid"])
    ]
    clean["midpoint"] = (clean["bid"] + clean["ask"]) / 2.0
    clean["relative_spread"] = (
        (clean["ask"] - clean["bid"]) / clean["midpoint"]
    )
    clean = clean[clean["relative_spread"] <= config.max_relative_spread]

    if "openInterest" in clean:
        clean = clean[
            clean["openInterest"].fillna(0) >= config.minimum_open_interest
        ]
    elif config.minimum_open_interest > 0:
        raise ValueError("openInterest is required by the selected filter")

    if "volume" in clean:
        clean = clean[clean["volume"].fillna(0) >= config.minimum_volume]
    elif config.minimum_volume > 0:
        raise ValueError("volume is required by the selected filter")

    return clean.sort_values("strike").reset_index(drop=True)


def analyze_option_chain(
    quotes: pd.DataFrame,
    *,
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
    filter_config: ChainFilterConfig | None = None,
) -> pd.DataFrame:
    """Recover implied volatility and Black-Scholes Greeks for valid quotes."""
    if spot <= 0 or time_to_expiry <= 0:
        raise ValueError("spot and time_to_expiry must be positive")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    clean = clean_option_quotes(quotes, filter_config)
    rows = []
    for _, quote in clean.iterrows():
        base = OptionInputs(
            spot=spot,
            strike=float(quote["strike"]),
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=0.20,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )
        lower, upper = option_bounds(base)
        midpoint = float(quote["midpoint"])
        if midpoint < lower - 1e-10 or midpoint > upper + 1e-10:
            continue
        try:
            iv = implied_volatility(midpoint, base)
            priced = OptionInputs(
                spot=spot,
                strike=base.strike,
                time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate,
                volatility=iv,
                dividend_yield=dividend_yield,
                option_type=option_type,
            )
            greeks = black_scholes_greeks(priced)
        except ValueError:
            continue

        rows.append(
            {
                **quote.to_dict(),
                "option_type": option_type,
                "time_to_expiry": float(time_to_expiry),
                "moneyness": float(base.strike / spot),
                "log_moneyness": float(np.log(base.strike / spot)),
                "implied_volatility": float(iv),
                **greeks,
            }
        )
    return pd.DataFrame(rows)


def volatility_smile(chain_analytics: pd.DataFrame) -> pd.DataFrame:
    """Return the core smile columns, sorted by moneyness."""
    required = {"strike", "moneyness", "implied_volatility"}
    missing = required.difference(chain_analytics.columns)
    if missing:
        raise ValueError(f"chain_analytics is missing: {sorted(missing)}")
    return chain_analytics[
        ["strike", "moneyness", "log_moneyness", "implied_volatility"]
    ].sort_values("moneyness").reset_index(drop=True)


def combine_expiration_analytics(
    analytics_by_expiration: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine several analyzed expirations into a long-form IV surface table."""
    rows = []
    for expiration, frame in analytics_by_expiration.items():
        if frame.empty:
            continue
        copy = frame.copy()
        copy["expiration"] = expiration
        rows.append(copy)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["expiration", "moneyness"]
    )
