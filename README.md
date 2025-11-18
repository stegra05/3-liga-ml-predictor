# 3. Liga Match Predictor ⚽️ 🇩🇪

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

**A production-grade machine learning pipeline for predicting German 3. Liga football matches.**

This project demonstrates a complete end-to-end ML system, from data ingestion and feature engineering to model training, evaluation, and deployment via a CLI. It addresses real-world challenges like data leakage prevention, temporal validation, and rigorous backtesting.

---

## 🚀 Key Features

*   **Robust Data Pipeline**: Aggregates data from 4+ sources (OpenLigaDB, FBref, OddsPortal, Transfermarkt) into a normalized SQLite database.
*   **Advanced Feature Engineering**: Implements Elo & Pi-ratings, recent form analysis, head-to-head stats, and live weather integration via Open-Meteo.
*   **Strict Backtesting**: Custom evaluation framework enforcing temporal separation to prevent data leakage (no "future peeking").
*   **Modern Tech Stack**: Built with **Poetry**, **Typer** (CLI), **SQLAlchemy**, **Pandas**, **Scikit-learn**, and **MLflow**.
*   **Reproducible Research**: Full experiment tracking and artifact management.

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.9+, Poetry |
| **Data** | SQLite, SQLAlchemy, Pandas, NumPy |
| **ML** | Scikit-learn, Random Forest, MLflow |
| **CLI** | Typer, Rich |
| **APIs** | OpenLigaDB, Open-Meteo |
| **Scraping** | Selenium, BeautifulSoup |

## 📊 Performance

*Realistic backtesting results after fixing historical data leakage:*

| Evaluation Mode | Accuracy | ROI % | Description |
|:---|---:|---:|:---|
| **Expanding Season** | **43.7%** | **-2.20%** | Train on all history, test on next season. |
| **Sliding Season** | **44.1%** | **+4.72%** | Train on 4-year window (tests concept drift). |
| *Baseline (Favorite)* | *47.1%* | *-* | Simple "bet on favorite" strategy. |

> **Note**: The primary goal is a robust, leak-free engineering framework. The current model is a baseline Random Forest without extensive hyperparameter tuning.

## ⚡️ Quick Start

### 1. Installation

```bash
# Clone and install dependencies
git clone https://github.com/stegra05/3-liga-ml-predictor.git
cd 3-liga-ml-predictor
poetry install
```

### 2. Setup

```bash
# Initialize database
poetry run liga-predictor db-init
```

### 3. Predict

```bash
# Predict the next upcoming matchday
poetry run liga-predictor
```

## 📂 Project Structure

```
├── data/               # Raw and processed datasets
├── database/           # SQLite database and schema
├── models/             # Serialized ML models
├── src/
│   └── liga_predictor/
│       ├── cli.py      # Command-line interface entry point
│       ├── data/       # Scrapers and data collectors
│       ├── features/   # Feature engineering logic
│       └── model/      # Training and evaluation pipelines
└── tests/              # Unit and integration tests
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
