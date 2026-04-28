# Crypto Statistical Arbitrage Research

## Overview

This repository contains the research component of a decoupled Crypto Statistical Arbitrage architecture.

It is focused on discovering multivariate cointegrated pairs using a data pipeline built with:
- PCA
- clustering
- graph community detection
- copula analysis
- reinforcement learning (PPO)

The execution layer is separate and is described by the overall architecture:
- Research: pipeline discovering multivariate cointegrated pairs via PCA, clustering, graphs, and RL (PPO); tracking copula/band strategies.
- Execution: ByBit mean-reversion prototype with live liquidity checks, kill-switch, and Kelly-Criterion position sizing.
- Stack: Python, PyTorch, Stable-Baselines3, NetworkX, Scikit-Learn, Statsmodels.

This repository is intended for research, strategy discovery, and signal generation rather than production trading.

## Project Goals

- Discover and rank statistical arbitrage candidate pairs in crypto markets.
- Filter liquid trading universes using ByBit data.
- Build stable pair selection and risk-aware candidate scoring.
- Generate analytics-ready outputs for backtesting and execution.

## Repo Structure

- `main_strategy.py` — research entrypoint and example pipeline flow.
- `config_strategy_api.py` — ByBit API configuration, testnet/mainnet toggle, and endpoint setup.
- `func_get_symbols.py` — fetches tradeable symbols and applies liquidity filtering.
- `func_price_klines.py` — fetches historical klines from ByBit.
- `func_prices_json.py` — stores price history to JSON for later processing.
- `func_cointegration.py` — cointegration tests, spread calculation, z-score generation, and pair selection.
- `func_pca_analysis.py` — PCA support functions for static and rolling analysis.
- `func_clustering.py` — symbol clustering using KMeans and DBSCAN.
- `func_graph_analysis.py` — graph construction and community detection for pair relationships.
- `func_copulas.py` — Gaussian and Student-t copula fitting and tail dependence.
- `func_plot_trends.py` — visualization tools for price, spread, z-score, and copula behavior.
- `func_reinforcement_learning.py` — PPO environment and ranking model for pair selection.
- `func_portfolio_optimization.py` — mean-variance optimizer for portfolio weight selection.
- `func_liquidity_filter.py` — liquidity screening based on volume, turnover, and open interest.
- `main_strategy_test_mine.ipynb` — notebook for experimentation and research snapshots.
- `extended_price_list.json` — additional saved price data.
- `professional_pairs-checkpoint.csv` — checkpointed output for pair research.

## Core Pipeline

1. **Symbol discovery**
   - Retrieve ByBit linear USDT tickers.
   - Filter symbols by liquidity, quote currency, trading status, and open interest.

2. **Price history collection**
   - Fetch historical klines via `func_price_klines.py`.
   - Store price series in `1_price_list.json` using `func_prices_json.py`.

3. **Cointegration discovery**
   - Use Engle-Granger cointegration via `statsmodels`.
   - Calculate hedge ratios, spreads, zero crossings, and z-score windows.

4. **Feature extraction and clustering**
   - Apply PCA to symbol return series.
   - Cluster candidates using KMeans or DBSCAN.

5. **Graph analysis**
   - Construct symbol graphs from cointegrated pair relationships.
   - Detect communities using Louvain clustering.

6. **Copula analysis**
   - Fit Gaussian and Student-t copulas to ranked returns.
   - Evaluate tail dependence and joint distribution structure.

7. **RL ranking**
   - Train a PPO agent to score and rank candidate pairs.
   - Use reward signals from pair attributes and composite scores.

8. **Portfolio optimization**
   - Use mean-variance optimization to compute portfolio weights.
   - Target risk-adjusted exposure for selected pair sets.

9. **Visualization and backtesting support**
   - Plot trends, spread, z-score, and copula structure.
   - Save backtest-ready CSV outputs.

## Usage

### Requirements

- Python 3.10+ recommended
- Core libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `statsmodels`
  - `scipy`
  - `matplotlib`
  - `networkx`
  - `python-louvain` or `community`
  - `stable-baselines3`
  - `gym`
  - `pybit`
  - `pykalman`
  - `kneed`

### Setup

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn statsmodels scipy matplotlib networkx python-louvain stable-baselines3 gym pybit pykalman kneed
```

2. Configure API credentials in `config_strategy_api.py`:
   - Set `testnet = True` for sandbox testing.
   - Replace `api_key_testnet` / `api_secret_testnet` with your keys.
   - Do not commit secrets to source control.

3. Run the research script:

```bash
python main_strategy.py
```

### Example Pipeline

`main_strategy.py` contains example steps for:
- fetching symbols
- storing prices
- finding cointegrated pairs
- plotting trends for a selected pair

Many research steps are currently commented to allow targeted experimentation.

## Output Files

- `1_price_list.json` — saved symbol price histories.
- `cointegrated_pairs_trial.csv` — discovered cointegrated pairs with statistics.
- `3_backtest_file.csv` — backtest-ready pair price spread and z-score data.
- `professional_pairs-checkpoint.csv` — candidate pair checkpoint data from research.

## Notes

- This repository is research-grade and intended for strategy development.
- The decoupled architecture separates strategy discovery from live execution.
- The execution bot is meant to be implemented independently using the research outputs.
- Use live funds only after extensive backtesting and proper risk controls.

## Research Narrative

This repository supports a broader project described as:

> Crypto Statistical Arbitrage Bot (Decoupled Architecture) Jan 2025 – Present
> Research: Pipeline discovering multivariate cointegrated pairs via PCA, clustering, graphs, and RL (PPO); tracking copula/band strategies.
> Research on GitHub: Statistical Arbitrage Strategy
> Execution: ByBit mean-reversion prototype with live liquidity checks, Kill-switch, and Kelly-Criterion position sizing.
> Stack: Python, PyTorch, Stable-Baselines3, NetworkX, Sklearn, Statsmodels.


## A few ss from the project
<img width="1062" height="1280" alt="photo_6307308677504897318_y" src="https://github.com/user-attachments/assets/ff8e06ad-2e05-4f53-b985-b29df3901db4" />

<img width="1080" height="5789" alt="Screenshot_20260429-015143 Chrome~2" src="https://github.com/user-attachments/assets/705e622e-0ebe-48c5-a764-57411c24ddbf" />

<img width="1080" height="954" alt="photo_6307308677504897317_y" src="https://github.com/user-attachments/assets/fa3d430d-e768-4bd0-8728-e3c985b12360" />




## License

Use this project for research, evaluation, and further development. Update licensing details as needed.
