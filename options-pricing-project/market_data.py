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
