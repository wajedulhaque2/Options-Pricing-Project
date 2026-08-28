import numpy as np
import pytest

from option_portfolio import (
    OptionLeg,
    expiration_value,
    portfolio_greeks,
    scenario_grid,
)


def test_call_vertical_has_bounded_expiration_value():
    legs = [
        OptionLeg(100, 0.5, 0.2, "call", 1),
        OptionLeg(110, 0.5, 0.2, "call", -1),
    ]
    values = expiration_value(legs, np.array([80, 100, 105, 110, 130]))
    assert values.max() == pytest.approx(1000.0)
    assert values.iloc[0] == 0


def test_vertical_delta_is_positive_but_smaller_than_long_call_delta():
    long_call = [OptionLeg(100, 0.5, 0.2, "call", 1)]
    spread = [
        OptionLeg(100, 0.5, 0.2, "call", 1),
        OptionLeg(110, 0.5, 0.2, "call", -1),
    ]
    long_delta = portfolio_greeks(long_call, 100, 0.03, 0)["delta"]
    spread_delta = portfolio_greeks(spread, 100, 0.03, 0)["delta"]
    assert 0 < spread_delta < long_delta


def test_zero_shock_scenario_has_zero_pnl():
    legs = [
        OptionLeg(100, 0.5, 0.25, "call", 1),
        OptionLeg(100, 0.5, 0.25, "put", 1),
    ]
    grid = scenario_grid(
        legs,
        100,
        [-0.1, 0, 0.1],
        [-0.1, 0, 0.1],
        risk_free_rate=0.03,
    )
    base = grid[
        (grid.spot_shock == 0) & (grid.volatility_shock == 0)
    ].iloc[0]
    assert base.pnl == pytest.approx(0, abs=1e-10)
    assert len(grid) == 9
