# Monte Carlo Based Option Pricing Model

## Live Deployments

### European Option Pricing App

[European Option Pricing App](https://montecarlooption-g8o9uw9f9cburuj23zjbaw.streamlit.app/)

Features included:

* Monte Carlo Simulation
* Black-Scholes Pricing
* Greeks Visualization
* GBM Path Simulation

---

### American & Barrier Option Pricing App

[American & Barrier Option Pricing App](https://americanbarrier-nu6uyaaja47gznizdmfaeb.streamlit.app/)

Features included:

* American Option Pricing
* LSMC (Least Squares Monte Carlo) Method
* Binomial Tree Method
* Barrier Options
* Asian Options 

---
## Project Overview

The Monte Carlo Based Option Pricing Model is a quantitative finance web application developed using Streamlit that enables users to simulate and price financial derivatives using stochastic simulations and numerical methods.

The platform supports pricing and analysis of:

* European Options
* American Options
* Exotic Options

using advanced quantitative finance techniques such as:

* Monte Carlo Simulation
* Black-Scholes Analytical Pricing
* Least Squares Monte Carlo (LSMC)
* Binomial Tree Methods

The project is designed for finance students, quantitative researchers, and developers interested in computational finance and derivative pricing.

---

# Features

## Real-Time Market Data

* Fetches live historical stock data using Yahoo Finance (`yfinance`)
* Supports NSE stocks such as `INFOSYS.NS`
* Automatically estimates volatility using historical log returns

---

# European Option Pricing

Supports:

* European Call Options
* European Put Options

## Pricing Methods

* Monte Carlo Simulation
* Black-Scholes Formula

## Greeks Calculation

Computes:

* Delta
* Gamma
* Vega
* Theta
* Rho

for analytical and simulation-based pricing.

---

# American Option Pricing

Supports:

* American Call Options
* American Put Options

## Numerical Methods

### Least Squares Monte Carlo (LSMC)

* Models optimal early exercise strategy
* Uses regression-based continuation value estimation

### Binomial Tree Method

* Recursive option valuation framework
* Supports early exercise decisions

---

# Exotic Option Pricing

Supports pricing of path-dependent derivatives.

## Asian Options

* Average price payoff calculation
* Monte Carlo-based pricing

## Barrier Options

Supports:

* Up-In Options
* Down-In Options

Barrier conditions are simulated dynamically using GBM price paths.

---

# Monte Carlo Simulation Engine

The application simulates stock prices using Geometric Brownian Motion (GBM).

## Geometric Brownian Motion (GBM)

The underlying stock price dynamics are modeled as:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t
$$

where:

- **$S_t$** = stock price at time $t$
- **$\mu$** = expected return (drift)
- **$\sigma$** = volatility
- **$W_t$** = Wiener process (Brownian motion)
- **$dW_t$** = infinitesimal increment of the Wiener process

---

# Interactive Visualizations

The application includes several visual analytics features:

* Simulated GBM stock price paths
* Monte Carlo convergence plots
* Greeks visualization
* Option price comparison charts
* Sensitivity analysis plots

---

# Streamlit User Interface

The project includes a clean interactive UI built with Streamlit.

## User Controls

Users can dynamically configure:

* Strike Percentage
* Time to Maturity (T)
* Risk-Free Interest Rate (r)
* Number of Simulations (M)
* Number of Timesteps (N)
* Volatility assumptions

---

# Tech Stack

| Category              | Technology  |
| --------------------- | ----------- |
| Programming Language  | Python 3.9+ |
| Frontend/UI           | Streamlit   |
| Numerical Computing   | NumPy       |
| Data Analysis         | Pandas      |
| Visualization         | Matplotlib  |
| Financial Data API    | yfinance    |
| Statistical Functions | SciPy       |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Nitya-sigadapu/monte_carlo-option.git
cd monte_carlo-option
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```
---

# How It Works

1. Pulls historical stock price data using Yahoo Finance
2. Estimates volatility using log returns
3. Simulates Geometric Brownian Motion stock paths
4. Computes discounted expected payoffs
5. Prices derivatives using:

   * Monte Carlo methods
   * Black-Scholes formula
   * LSMC
   * Binomial Trees
6. Computes Greeks and sensitivity metrics
7. Displays visualizations and pricing outputs

---

# Sample Results

# European Options

## Monte Carlo Method

| Metric            | Value   |
| ----------------- | ------- |
| Put Option Price  | ₹195.88 |
| Call Option Price | ₹82.62  |

## Greeks (Call Option)

| Greek | Value   |
| ----- | ------- |
| Delta | 0.41    |
| Gamma | -0.00   |
| Vega  | 452.87  |
| Theta | -158.50 |
| Rho   | 288.53  |

---

## Black-Scholes Model

| Metric            | Value   |
| ----------------- | ------- |
| Put Option Price  | ₹82.73  |
| Call Option Price | ₹199.86 |

## Greeks

| Greek | Value   |
| ----- | ------- |
| Delta | 0.40    |
| Gamma | 0.00    |
| Vega  | 439.66  |
| Theta | -152.94 |
| Rho   | 281.31  |

---

# Future Improvements

* GPU-accelerated simulations
* Variance reduction techniques
* Heston stochastic volatility model integration
* Implied volatility surface visualization
* Greeks heatmaps and 3D visualizations
* Multi-asset option pricing
* Portfolio risk analysis dashboard
* Unit testing and CI/CD integration

---

# References

1. Quant Guild
   https://youtu.be/-1RYvajksjQ

2. Very Normal
   https://youtu.be/Cb-GwN6jhNE

3. Option Pricing Models Repository
   https://github.com/just-krivi/option-pricing-models

4. Monte Carlo Pricing Tutorial
   https://youtu.be/OdWLP8umw3A

5. Quantitative Finance Tutorial
   https://youtu.be/r7cn3WS5x9c

---

# Authors

* Nitya
  https://github.com/Nitya-sigadapu

* Charan
  https://github.com/thchrn

* Prayuktha
  https://github.com/Prayuktha-Lucky-Reddy

---

# License

This project is licensed under the MIT License.

---

# Contributions

Contributions, suggestions, and pull requests are welcome.

Feel free to fork the repository and improve the project.
