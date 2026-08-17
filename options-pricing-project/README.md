# Options Pricing Project — Implementation Directory

For the project overview, model comparison, verified calculations and portfolio walkthrough, see the [root README](../README.md).

This directory contains the pricing models, interactive application, notebooks, tests and saved outputs behind the project.

## Key files

- `main.py` — reproducible pricing/risk workflow and output generation.
- `main.ipynb` — notebook analysis and visual exploration.
- `complete_options_pricing_theory_guide.ipynb` — detailed options-pricing theory.
- `app.py` — Streamlit calculator.
- `black_scholes.py` — analytical pricing, Greeks, bounds and parity checks.
- `binomial.py` — CRR European/American tree valuation.
- `monte_carlo.py` — simulation-based pricing and uncertainty estimates.
- `implied_volatility.py` — implied-volatility solver.
- `analytics.py` — model comparison and sensitivity analysis.
- `tests/` — automated model and validation tests.
- `outputs/` — saved model comparison, calculation summary, price surface and Greek profiles used in the root README.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Interactive calculator:

```bash
streamlit run app.py
```

Automated tests:

```bash
pytest
```
