"""Interactive options pricing, volatility and strategy-risk application."""

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
from option_portfolio import expiration_value, portfolio_greeks, scenario_grid
from strategy_templates import build_strategy
from volatility_surface import ChainFilterConfig, analyze_option_chain

st.set_page_config(page_title="Options Volatility & Strategy Research", layout="wide")
st.title("Options Volatility & Strategy Research")
st.caption(
    "Pricing models, Greeks, implied volatility, market-chain cleaning, "
    "multi-leg risk and spot/volatility scenario analysis."
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
    simulations = st.slider(
        "Monte Carlo simulations",
        10_000,
        500_000,
        100_000,
        10_000,
    )

inputs = OptionInputs(
    spot=float(spot),
    strike=float(strike),
    time_to_expiry=float(time_to_expiry),
    risk_free_rate=float(risk_free_rate),
    volatility=float(volatility),
    dividend_yield=float(dividend_yield),
    option_type=option_type,
)

(
    calculator_tab,
    greeks_tab,
    iv_tab,
    surface_tab,
    american_tab,
    strategy_tab,
    market_tab,
) = st.tabs(
    [
        "Calculator",
        "Greeks",
        "Implied volatility",
        "Scenario surface",
        "American exercise",
        "Strategy risk",
        "Live market IV",
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

with strategy_tab:
    st.subheader("Multi-leg portfolio risk")
    template = st.selectbox(
        "Strategy template",
        ["Long option", "Vertical spread", "Long straddle"],
    )
    width = st.number_input(
        "Vertical width",
        min_value=0.01,
        value=max(float(inputs.strike) * 0.10, 1.0),
        step=1.0,
        disabled=template != "Vertical spread",
    )

    try:
        legs = build_strategy(template, inputs, float(width))
        leg_table = pd.DataFrame(
            [
                {
                    "option_type": leg.option_type,
                    "strike": leg.strike,
                    "quantity": leg.quantity,
                    "volatility": leg.volatility,
                    "time_to_expiry": leg.time_to_expiry,
                    "multiplier": leg.contract_multiplier,
                }
                for leg in legs
            ]
        )
        st.dataframe(leg_table, use_container_width=True, hide_index=True)

        aggregate_greeks = portfolio_greeks(
            legs,
            inputs.spot,
            inputs.risk_free_rate,
            inputs.dividend_yield,
        )
        st.dataframe(
            pd.Series(aggregate_greeks, name="portfolio_value").to_frame(),
            use_container_width=True,
        )

        spot_shocks = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]
        volatility_shocks = [-0.10, -0.05, 0.0, 0.05, 0.10]
        scenarios = scenario_grid(
            legs,
            base_spot=inputs.spot,
            spot_shocks=spot_shocks,
            volatility_shocks=volatility_shocks,
            risk_free_rate=inputs.risk_free_rate,
            dividend_yield=inputs.dividend_yield,
        )
        pnl_grid = scenarios.pivot(
            index="spot_shock",
            columns="volatility_shock",
            values="pnl",
        )
        pnl_grid.index = [f"{value:+.0%}" for value in pnl_grid.index]
        pnl_grid.columns = [f"{value:+.0%}" for value in pnl_grid.columns]
        st.markdown("**Scenario P&L: spot shock × absolute volatility shock**")
        st.dataframe(pnl_grid.round(2), use_container_width=True)

        expiry_spots = np.linspace(inputs.spot * 0.5, inputs.spot * 1.5, 151)
        payoff = expiration_value(legs, expiry_spots)
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(payoff.index, payoff.values)
        axis.axhline(0.0, linestyle="--")
        axis.axvline(inputs.spot, linestyle=":")
        axis.set_xlabel("Underlying price at expiry")
        axis.set_ylabel("Portfolio expiration value")
        axis.set_title(f"{template}: expiration value")
        axis.grid(True)
        st.pyplot(figure)
    except ValueError as error:
        st.error(str(error))

with market_tab:
    st.subheader("Clean quoted chains and recover market implied volatility")
    ticker = st.text_input("Ticker", value="AAPL").upper()
    max_relative_spread = st.slider(
        "Maximum bid/ask spread as % of midpoint",
        min_value=0.05,
        max_value=1.00,
        value=0.35,
        step=0.05,
    )
    minimum_open_interest = st.number_input(
        "Minimum open interest",
        min_value=0,
        value=0,
        step=10,
    )

    if st.button("Load expirations"):
        try:
            st.session_state["market_ticker"] = ticker
            st.session_state["expirations"] = option_expirations(ticker)
        except Exception as error:
            st.error(str(error))

    expirations = st.session_state.get("expirations", ())
    if expirations:
        expiration = st.selectbox("Expiration", expirations)
        if st.button("Load and analyze option chain"):
            try:
                chain = download_option_chain(ticker, expiration)
                expiry_date = pd.Timestamp(expiration).normalize()
                today = pd.Timestamp.today().normalize()
                calendar_days = max(int((expiry_date - today).days), 1)
                market_time = calendar_days / 365.0
                filter_config = ChainFilterConfig(
                    max_relative_spread=float(max_relative_spread),
                    minimum_open_interest=int(minimum_open_interest),
                )

                call_analytics = analyze_option_chain(
                    chain.calls,
                    spot=chain.spot,
                    time_to_expiry=market_time,
                    risk_free_rate=inputs.risk_free_rate,
                    dividend_yield=inputs.dividend_yield,
                    option_type="call",
                    filter_config=filter_config,
                )
                put_analytics = analyze_option_chain(
                    chain.puts,
                    spot=chain.spot,
                    time_to_expiry=market_time,
                    risk_free_rate=inputs.risk_free_rate,
                    dividend_yield=inputs.dividend_yield,
                    option_type="put",
                    filter_config=filter_config,
                )

                cols = st.columns(3)
                cols[0].metric(f"{ticker} spot", f"{chain.spot:,.2f}")
                cols[1].metric("Calendar days to expiry", str(calendar_days))
                try:
                    cols[2].metric(
                        "One-year historical volatility",
                        f"{historical_volatility(ticker):.2%}",
                    )
                except Exception:
                    cols[2].metric("One-year historical volatility", "Unavailable")

                calls_tab, puts_tab = st.tabs(["Calls", "Puts"])
                for tab, analytics_frame, label in (
                    (calls_tab, call_analytics, "Call"),
                    (puts_tab, put_analytics, "Put"),
                ):
                    with tab:
                        if analytics_frame.empty:
                            st.warning(
                                "No contracts survived the selected quote/liquidity "
                                "filters and implied-volatility checks."
                            )
                            continue
                        display_columns = [
                            "contractSymbol",
                            "strike",
                            "bid",
                            "ask",
                            "midpoint",
                            "relative_spread",
                            "openInterest",
                            "volume",
                            "moneyness",
                            "implied_volatility",
                            "delta",
                            "gamma",
                            "vega_per_1pct",
                            "theta_per_day",
                        ]
                        available = [
                            column
                            for column in display_columns
                            if column in analytics_frame.columns
                        ]
                        st.dataframe(
                            analytics_frame[available],
                            use_container_width=True,
                            hide_index=True,
                        )
                        figure, axis = plt.subplots(figsize=(10, 5))
                        axis.plot(
                            analytics_frame["strike"],
                            analytics_frame["implied_volatility"],
                            marker="o",
                        )
                        axis.axvline(chain.spot, linestyle="--", label="Spot")
                        axis.set_xlabel("Strike")
                        axis.set_ylabel("Implied volatility")
                        axis.set_title(f"{ticker} {label} implied-volatility smile")
                        axis.grid(True)
                        axis.legend()
                        st.pyplot(figure)
            except Exception as error:
                st.error(str(error))
