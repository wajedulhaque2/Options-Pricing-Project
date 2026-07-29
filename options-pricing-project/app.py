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
