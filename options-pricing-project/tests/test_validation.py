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
