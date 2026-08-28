import pandas as pd
import pytest

from black_scholes import OptionInputs, black_scholes_price
from volatility_surface import (
    ChainFilterConfig,
    analyze_option_chain,
    clean_option_quotes,
)


def test_quote_cleaner_removes_zero_bid_and_wide_spread():
    quotes = pd.DataFrame(
        {
            "strike": [90, 100, 110],
            "bid": [0, 5, 1],
            "ask": [2, 5.5, 3],
            "volume": [10, 10, 10],
            "openInterest": [100, 100, 100],
        }
    )
    clean = clean_option_quotes(
        quotes, ChainFilterConfig(max_relative_spread=0.25)
    )
    assert clean["strike"].tolist() == [100]


def test_chain_analysis_recovers_planted_implied_volatility():
    rows = []
    for strike in [90, 100, 110]:
        inputs = OptionInputs(100, strike, 0.5, 0.03, 0.25, 0.01, "call")
        price = black_scholes_price(inputs)
        rows.append(
            {
                "strike": strike,
                "bid": price - 0.02,
                "ask": price + 0.02,
                "volume": 50,
                "openInterest": 500,
            }
        )

    result = analyze_option_chain(
        pd.DataFrame(rows),
        spot=100,
        time_to_expiry=0.5,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        option_type="call",
        filter_config=ChainFilterConfig(max_relative_spread=0.5),
    )
    assert len(result) == 3
    assert result["implied_volatility"].tolist() == pytest.approx(
        [0.25, 0.25, 0.25], abs=1e-6
    )
    assert result["delta"].between(0, 1).all()
