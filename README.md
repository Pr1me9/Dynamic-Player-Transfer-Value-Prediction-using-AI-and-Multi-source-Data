# ⚽ TransferIQ

> AI-powered football player transfer value prediction using multi-source data, deep learning, and ensemble machine learning.

## Overview

TransferIQ predicts future football player market values by combining:

- 📊 Match performance data (StatsBomb)
- 💰 Historical market values (Transfermarkt)
- 🏥 Injury history
- 💬 Social sentiment analysis
- 📈 Time-series forecasting

The project leverages LSTM-based forecasting models and XGBoost ensembles to generate data-driven player valuation predictions.

---

## Key Features

- Multi-source football analytics pipeline
- Univariate & Multivariate LSTM forecasting
- Encoder-Decoder (Seq2Seq) LSTM models
- XGBoost ensemble learning
- Hyperparameter optimization dashboard
- Injury and sentiment-aware predictions
- Interactive Streamlit applications
- Multi-step transfer value forecasting

---

## Data Sources

| Source | Purpose |
|----------|----------|
| StatsBomb Open Data | Match events & player performance |
| Transfermarkt | Market values & transfer history |
| Injury Datasets | Injury records & recovery metrics |
| Reddit/Social Data | Sentiment analysis |

---

## ML Pipeline

```text
Data Collection
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
LSTM Forecasting
       ↓
XGBoost Ensemble
       ↓
Transfer Value Prediction
```

---

## Models

### Univariate LSTM
Forecasts future player value using historical market values.

### Multivariate LSTM
Uses additional features such as:

- Injuries
- Sentiment scores
- Minutes played
- Shots per 90
- Pressures per 90
- Cards per match

### Encoder-Decoder LSTM
Performs multi-step transfer value forecasting.

### XGBoost Ensemble
Combines deep learning forecasts with gradient boosting for improved accuracy.

---

## Tech Stack

**Data Engineering**
- Python
- Pandas
- MySQL

**Machine Learning**
- TensorFlow / Keras
- Scikit-Learn
- XGBoost

**Data Collection**
- Selenium
- BeautifulSoup
- Requests

**Visualization & Deployment**
- Plotly
- Matplotlib
- Streamlit

---

## Results

12M+ StatsBomb event records processed

1,500+ players analyzed

Multi-source feature engineering pipeline

Ensemble forecasting architecture

Interactive prediction dashboards

---


## Installation

```bash
git clone https://github.com/Pr1me9/Dynamic-Player-Transfer-Value-Prediction-using-AI-and-Multi-source-Data.git

```

---

## Future Enhancements

- Transformer-based forecasting
- Live football API integration
- Explainable AI (SHAP)
- Player similarity engine
- Injury risk prediction

---