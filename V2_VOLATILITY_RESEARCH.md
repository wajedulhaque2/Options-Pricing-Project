# Options V2: Volatility & Strategy Research

This upgrade extends the project from a model calculator into a market-oriented derivatives research toolkit while preserving the original Black-Scholes, binomial, Monte Carlo and Streamlit workflows.

## Option-chain analytics

`options-pricing-project/volatility_surface.py` cleans quoted option chains before implied volatility is calculated. It rejects non-positive bids, crossed markets and quotes whose bid/ask spread exceeds a configurable percentage of the midpoint. Optional volume and open-interest thresholds can also be applied.

```python
from volatility_surface import ChainFilterConfig, analyze_option_chain

calls = analyze_option_chain(
    chain.calls,
    spot=chain.spot,
    time_to_expiry=days_to_expiry / 365.0,
    risk_free_rate=0.04,
    dividend_yield=0.0,
    option_type="call",
    filter_config=ChainFilterConfig(
        max_relative_spread=0.25,
        minimum_open_interest=100,
    ),
)
```

Each surviving contract receives midpoint, moneyness, log-moneyness, implied volatility and analytical Black-Scholes Greeks. `volatility_smile()` produces a strike/moneyness smile table, while `combine_expiration_analytics()` creates a long-form table suitable for plotting an implied-volatility surface across expirations.

## Generic multi-leg portfolios

`options-pricing-project/option_portfolio.py` represents strategies as combinations of generic `OptionLeg` objects instead of hard-coding every named strategy.

```python
from option_portfolio import OptionLeg, portfolio_greeks, scenario_grid

call_spread = [
    OptionLeg(100, 0.50, 0.25, "call", quantity=1),
    OptionLeg(110, 0.50, 0.25, "call", quantity=-1),
]

risk = portfolio_greeks(call_spread, spot=100, risk_free_rate=0.04)
stress = scenario_grid(
    call_spread,
    base_spot=100,
    spot_shocks=[-0.10, 0.0, 0.10],
    volatility_shocks=[-0.10, 0.0, 0.10],
    risk_free_rate=0.04,
)
```

This supports calls, puts, long and short legs, arbitrary quantities, contract multipliers, expiration payoff profiles, aggregate Greeks and spot/volatility scenario P&L. Named strategies such as verticals, straddles, strangles, butterflies and condors are therefore compositions of the same reusable primitives.

## Monte Carlo uncertainty correction

The original simulation already used antithetic variates and a control variate. V2 fixes a subtle uncertainty-estimation issue: antithetic `z` and `-z` payoffs are correlated by construction, so they must not be treated as independent observations when calculating the standard error.

The revised implementation first averages each antithetic payoff pair and then estimates the sampling standard error from those independent pair averages. Antithetic mode now requires an even simulation count so every draw has an exact partner.

## Verification scenarios

The V2 tests include the following model and risk checks:

- synthetic Black-Scholes quotes generated at 25% volatility recover approximately 25% implied volatility across several strikes;
- zero-bid and excessively wide quotes are rejected before IV inversion;
- a long 100 / short 110 call vertical with one contract and a 100x multiplier has a maximum expiration value of $1,000;
- the vertical's delta is positive but lower than the standalone long call's delta;
- a zero spot/volatility shock produces zero scenario P&L;
- the antithetic Monte Carlo standard error exactly matches an independent manual calculation from payoff-pair averages.

GitHub Actions runs these checks together with the pre-existing Black-Scholes, binomial, Monte Carlo, implied-volatility and validation tests on every push and pull request.

## Research scope

The repository now supports **current-market volatility and strategy analysis**, but it does not pretend to contain a historical options-strategy backtest. Yahoo Finance supplies current option chains and underlying history rather than a reliable point-in-time archive of complete historical chains. A genuine implied-versus-realized-volatility trading backtest should therefore be added only when point-in-time historical option data is available, rather than reconstructing history from today's chain.
