# Options Pricing Calculator

A complete mathematical derivatives-pricing project covering:

- Black-Scholes-Merton European call and put pricing
- Continuous dividend yield
- Analytical Greeks: delta, gamma, vega, theta, and rho
- Put-call parity and no-arbitrage bounds
- Implied volatility using SciPy's `brentq` root solver
- Cox-Ross-Rubinstein binomial trees
- European and American exercise
- Risk-neutral Monte Carlo simulation
- Antithetic and control variates
- Confidence intervals and standard errors
- Price surfaces and Greek profiles
- Optional Yahoo Finance option chains
- Interactive Streamlit calculator
- Automated tests
- A complete theory notebook

## Project structure

```text
options-pricing-project/
├── black_scholes.py
├── implied_volatility.py
├── binomial.py
├── monte_carlo.py
├── analytics.py
├── market_data.py
├── main.py
├── main.ipynb
├── app.py
├── complete_options_pricing_theory_guide.ipynb
├── requirements.txt
├── tests/
└── outputs/
```

## Windows setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the complete workflow

```powershell
python main.py
```

The workflow exports model comparisons, a price surface, Greek profiles, a JSON summary, and charts to `outputs/`.

## Run the interactive calculator

```powershell
streamlit run app.py
```

## Run the tests

```powershell
pytest -q
```

## Notebook use

Open `main.ipynb` and run all cells.

Open `complete_options_pricing_theory_guide.ipynb` for the full mathematical explanation.

## Important scope

Black-Scholes-Merton assumes a frictionless market, continuous trading, lognormal prices, constant volatility and rates, and European exercise. The tree relaxes the European-exercise restriction. Monte Carlo estimates European values numerically.

This project is educational and is not trading advice.
