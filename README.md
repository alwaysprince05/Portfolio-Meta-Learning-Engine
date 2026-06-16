# Portfolio Meta-Learning Engine

> **Author:** Prince Maurya  
> **License:** MIT  
> **Stack:** Python · Streamlit · Plotly · scikit-learn · SciPy · yfinance

---

## Overview

The **Portfolio Meta-Learning Engine** is an institutional-grade quantitative dashboard for adaptive portfolio construction. It dynamically selects the best portfolio strategy based on the current detected market regime using a **Bayesian meta-learning** approach, and visualizes market structure through **interactive 3D spectral decomposition**.

Built for quant researchers, algo traders, and finance enthusiasts who need more than a static allocation model.

---

## Features

| Feature | Description |
|---|---|
| 🧠 **Meta-Learning Engine** | Adapts portfolio strategies (Mean-Variance, Risk Parity, Momentum Tilt, etc.) to changing market regimes in real time |
| 📡 **Market Regime Detection** | Uses Gaussian Mixture Models (GMM) + PCA to detect volatility, correlation, and trend regimes |
| 📊 **Spectral Decomposition** | FFT-based rolling spectral surface visualized as an interactive 3D plot |
| 🎯 **Bayesian Strategy Selection** | Picks the highest Sharpe-ratio strategy per regime window with confidence scoring |
| ⚖️ **Adaptive Rebalancing** | Triggers rebalancing only on regime shifts — reducing unnecessary turnover |
| 📈 **Performance Analytics** | Equity curve, rolling Sharpe ratio, and max drawdown — all rendered interactively |
| 🌐 **3D Allocation Evolution** | Visualizes how asset weights shift across time in a 3D scatter plot |

---

## Strategies Supported

- **Mean-Variance** — Classic Markowitz optimization with risk aversion tuning
- **Risk Parity** — Equal risk contribution across all assets
- **Minimum Variance** — Lowest portfolio volatility allocation
- **Momentum Tilt** — Weights assets by Sharpe-adjusted momentum
- **Defensive Allocation** — Inverse-volatility weighting for capital preservation

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/princemaurya05/Portfolio-Meta-Learning-Engine.git
cd Portfolio-Meta-Learning-Engine
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy plotly yfinance scipy scikit-learn
```

### 3. Launch the dashboard
```bash
streamlit run portfolio_meta_learning_engine.py
```

### 4. Open in browser
The dashboard will open automatically at `http://localhost:8501`

---

## Project Structure

```
Portfolio-Meta-Learning-Engine/
│
├── portfolio_meta_learning_engine.py   # Core engine — all logic & visualizations
├── README.md                           # Project documentation
└── LICENSE                             # MIT License
```

---

## How It Works

```
Market Data (yfinance)
        │
        ▼
Feature Computation (returns, vol, correlation, trend)
        │
        ▼
Regime Detection (GMM + PCA + Trend Strength)
        │
        ▼
Bayesian Strategy Selection (Sharpe-scored per window)
        │
        ▼
Adaptive Rebalancing (triggered on regime change)
        │
        ▼
Performance + Spectral Analysis (FFT, 3D Surface)
        │
        ▼
Streamlit Dashboard (Interactive Plotly Charts)
```

---

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- plotly
- yfinance
- scipy
- scikit-learn

---

## Configuration (Sidebar Controls)

| Parameter | Range | Default | Description |
|---|---|---|---|
| Strategy Universe | Multiselect | MV, RP, MT | Strategies to include in the meta-learner |
| Lookback Window | 30–252 days | 90 | Rolling window for feature computation |
| Risk Aversion | 0.1–5.0 | 2.0 | Controls risk tolerance in Mean-Variance optimization |
| Regime Sensitivity | 0.1–2.0 | 1.0 | Threshold sensitivity for trend regime detection |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

Copyright © 2026 **Prince Maurya**

---

## Contact

For questions, suggestions, or collaborations — reach out via GitHub.
