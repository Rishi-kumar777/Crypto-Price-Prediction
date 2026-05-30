# Cryptocurrency Price Prediction using Hybrid LSTM–XGBoost

A hybrid forecasting model that combines LSTM-based temporal feature extraction with XGBoost regression for Bitcoin price prediction.

## Overview

This project uses 4 years of BTC-USD market data (2020–2024) covering major market regimes including the COVID-19 crash, the 2021 bull run, the 2022 bear market, and the 2023 consolidation phase.

The model follows a two-stage approach:

1. LSTM learns temporal patterns from historical market data.
2. XGBoost uses learned latent representations to generate final price predictions.

## Features

* Open, High, Low, Close, Volume
* MarketCap Proxy (Close × Volume)
* SMA(10)
* EMA(10)
* Rolling Volatility

## Model Architecture

LSTM(128) → LSTM(64) → LSTM(64) → Dense(64)

↓

XGBoost Regressor

* 120-day lookback window
* 64-dimensional latent feature embeddings
* RandomizedSearchCV hyperparameter tuning
* 60 cross-validated XGBoost training runs

## Results

| Metric | Score  |
| ------ | ------ |
| RMSE   | 0.0246 |
| MAPE   | 5.05%  |

The trained model also generates 7-day ahead cryptocurrency price forecasts.

## Tech Stack

Python • TensorFlow/Keras • XGBoost • Scikit-Learn • Pandas • NumPy • yFinance

## Research Inspiration

This implementation was inspired by:

**Crypto Price Prediction Using LSTM+XGBoost** by Mehul Gautam et al. (2025). The project adopts the paper's hybrid learning concept while implementing an independent BTC-USD forecasting pipeline.

## Future Improvements

* Additional technical indicators (RSI, MACD, Bollinger Bands)
* Sentiment-based features
* SHAP explainability
* Transformer-based architectures
* Walk-forward validation
