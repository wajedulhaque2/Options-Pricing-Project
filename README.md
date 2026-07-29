# Options Pricing and Risk Analytics Calculator

An interactive Python project for exploring option valuation, risk sensitivities and the assumptions behind common derivatives-pricing models.

## Overview

This project compares three widely used approaches to option pricing:

* Black–Scholes–Merton
* Cox–Ross–Rubinstein binomial trees
* Monte Carlo simulation

It also explores how option values respond to changes in the underlying asset price, volatility, time to expiry, interest rates and dividends.

The focus is not only on calculating a final price, but also on understanding why the models work, when they differ and how traders measure option risk.

## Start Here

Readers who are mainly interested in the theory and results should begin with:

1. **`complete_options_pricing_theory_guide.ipynb`**
   Introduces option payoffs, no-arbitrage bounds, risk-neutral pricing, Black–Scholes, binomial trees, Monte Carlo simulation, Greeks and implied volatility.

2. **`main.ipynb`**
   Demonstrates the pricing models, compares their outputs and visualises option values and risk sensitivities.

3. **`app.py`**
   Runs the interactive Streamlit calculator.

The remaining Python files contain the individual pricing models and analytics functions used by the notebook and application.

## Key Features

* European call and put pricing
* Black–Scholes analytical valuation
* European and American binomial-tree pricing
* Monte Carlo pricing with variance-reduction techniques
* Delta, gamma, vega, theta and rho calculations
* Implied-volatility estimation
* No-arbitrage bounds and parity checks
* Price and Greek sensitivity analysis
* American early-exercise analysis
* Optional market-data and option-chain integration
* Interactive Streamlit interface

## Purpose

The project is designed to make options theory more intuitive by connecting the mathematics to practical calculations and visualisations.

It is intended for education, quantitative-finance research and portfolio demonstration. It is not a recommendation to trade options or rely on any model without considering its assumptions and limitations.
