# Time-Series Forecasting Platform

A production-grade framework for building, evaluating, benchmarking, and deploying forecasting models ranging from classical statistical approaches to modern foundation models.

---

## Overview

This repository provides a unified framework for:

- Time-series preprocessing
- Feature engineering
- Forecast model training
- Backtesting and evaluation
- Hyperparameter optimization
- Experiment tracking
- API serving
- Model monitoring
- Production deployment

Supported domains include:

- Financial Markets
- Demand Forecasting
- Retail Analytics
- Energy Forecasting
- IoT Sensor Forecasting
- Economic Forecasting
- Supply Chain Planning

---

---

# Recommended Models by Use Case

Selecting the right forecasting model depends on data volume, forecast horizon, seasonality, feature availability, and business requirements.

| Use Case | Recommended Models | Why |
|-----------|--------------------|------|
| Sales Forecasting | Prophet, XGBoost | Handles seasonality, promotions, and business trends effectively |
| Demand Planning | XGBoost, LightGBM | Strong performance with multivariate features and external drivers |
| Stock Market Forecasting | LSTM, Transformer | Captures temporal dependencies and complex market behavior |
| Energy Load Forecasting | Informer, Autoformer | Designed for long-horizon forecasting and large-scale time-series |
| Retail Forecasting | Prophet + XGBoost | Combines trend/seasonality modeling with machine learning features |
| IoT Sensor Forecasting | TCN, GRU | Efficiently models sensor streams and temporal patterns |
| Financial Forecasting | PatchTST, Informer | State-of-the-art transformer architectures for financial time-series |
| Supply Chain Forecasting | Prophet, XGBoost, LightGBM | Handles seasonality, demand fluctuations, and external factors |
| Economic Forecasting | ARIMA, Prophet, Transformer | Effective for trend analysis and macroeconomic indicators |
| Weather Forecasting | LSTM, Transformer, Autoformer | Captures long-term dependencies and nonlinear patterns |
| Traffic Forecasting | GRU, TCN, Informer | Suitable for high-frequency sequential data |
| Inventory Forecasting | Holt-Winters, Prophet, XGBoost | Strong performance for seasonal demand planning |
| Large Scale Enterprise Forecasting | Hybrid Ensemble | Combines statistical, ML, and deep learning models for robustness |

---

# Model Selection Guide

### If your dataset is small (<10K observations)

Recommended:

- Naive
- Moving Average
- ARIMA
- Holt-Winters
- Prophet

### If your dataset is medium (10K–1M observations)

Recommended:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

### If your dataset is large (>1M observations)

Recommended:

- LSTM
- GRU
- TCN
- Informer
- Autoformer
- PatchTST

### If you need explainability

Recommended:

- ARIMA
- Prophet
- Random Forest
- XGBoost

### If you need the highest accuracy

Recommended:

- PatchTST
- Informer
- Autoformer
- TimeGPT
- Hybrid Ensemble Models

### If you need zero-shot forecasting

Recommended:

- TimeGPT
- Chronos
- Moirai
- TimesFM

---

# Recommended Learning Path

For practitioners new to forecasting, follow this progression:

1. Naive Forecasting
2. Moving Average
3. ARIMA
4. Holt-Winters
5. Prophet
6. XGBoost
7. LightGBM
8. LSTM
9. TCN
10. Transformer
11. Informer
12. Autoformer
13. PatchTST
14. TimeGPT / Chronos / Moirai

This progression moves from classical statistical methods to state-of-the-art foundation models while building intuition about time-series behavior and forecasting techniques.

---

# Forecasting Model Categories

The repository organizes forecasting algorithms into five major categories.

---

## 1. Baseline & Classical Statistical Models

Suitable for:

- Stable trends
- Linear patterns
- Univariate forecasting

Implemented Models:

| Model | Description |
|---------|-------------|
| Naive | Last observation forecasting |
| Moving Average | Rolling average prediction |
| Autoregressive (AR) | Lag-based forecasting |
| ARIMA | Autoregressive Integrated Moving Average |

Directory:

```text
src/models/baseline/
```

---

## 2. Exponential Smoothing Models

Suitable for:

- Recent trend emphasis
- Seasonal data
- Business forecasting

Implemented Models:

| Model | Description |
|---------|-------------|
| SES | Simple Exponential Smoothing |
| Holt | Trend Forecasting |
| Holt-Winters | Trend + Seasonality |

Directory:

```text
src/models/exponential_smoothing/
```

---

## 3. Decomposition-Based Models

Suitable for:

- Multiple seasonality
- Holiday effects
- Business KPIs

Implemented Models:

| Model | Description |
|---------|-------------|
| Prophet | Meta Prophet Forecasting |
| STL Forecast | Seasonal-Trend Decomposition |
| Seasonal Decompose | Classical Decomposition |

Directory:

```text
src/models/decomposition/
```

---

## 4. Machine Learning Regressors

Suitable for:

- Multivariate forecasting
- Nonlinear relationships
- External regressors

Implemented Models:

| Model | Description |
|---------|-------------|
| Random Forest | Ensemble Tree Regressor |
| XGBoost | Gradient Boosting |
| LightGBM | Fast Gradient Boosting |
| CatBoost | Categorical Boosting |
| SVR | Support Vector Regression |

Directory:

```text
src/models/machine_learning/
```

---

## 5. Deep Learning Models

Suitable for:

- Long sequence modeling
- Complex temporal dependencies

Implemented Models:

| Model | Description |
|---------|-------------|
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| TCN | Temporal Convolution Network |
| N-BEATS | Deep Residual Forecasting |

Directory:

```text
src/models/deep_learning/
```

---

## 6. Transformer-Based Models

Suitable for:

- Long horizon forecasting
- Large-scale datasets

Implemented Models:

| Model | Description |
|---------|-------------|
| Transformer | Vanilla Transformer |
| Informer | Efficient Long Sequence Forecasting |
| Autoformer | Auto-Correlation Transformer |
| PatchTST | Patch-based Time-Series Transformer |
| TimesFM | Google Time-Series Foundation Model |

Directory:

```text
src/models/transformers/
```

---

## 7. Foundation Models

Suitable for:

- Zero-shot forecasting
- Cross-domain forecasting
- Large-scale enterprise forecasting

Implemented Models:

| Model | Description |
|---------|-------------|
| TimeGPT | Nixtla Foundation Model |
| Chronos | Amazon Foundation Model |
| Moirai | Universal Time-Series Foundation Model |

Directory:

```text
src/models/foundation_models/
```

---

# Feature Engineering

Supported Features:

- Lag Features
- Rolling Statistics
- Moving Averages
- Trend Features
- Seasonality Features
- Calendar Features
- Holiday Features
- Fourier Features

Directory:

```text
src/features/
```

---

# Evaluation Metrics

Implemented Metrics:

- MAE
- RMSE
- MAPE
- SMAPE
- WAPE
- MASE
- R²

Directory:

```text
src/evaluation/
```

---

# Backtesting

Supported Strategies:

- Expanding Window Validation
- Rolling Window Validation
- Walk Forward Validation

Directory:

```text
src/evaluation/backtesting.py
```

---

# Experiment Tracking

Supported Platforms:

- MLflow
- Weights & Biases (WandB)

Directory:

```text
experiments/
```

---

# API Deployment

FastAPI inference server:

```bash
uvicorn src.serving.api:app --reload
```

API Endpoint:

```http
POST /forecast
```

Request:

```json
{
  "series": [100, 105, 110, 112]
}
```

Response:

```json
{
  "forecast": [115, 118, 120]
}
```

---

# Training

Train a model:

```bash
python scripts/train.py \
  --model xgboost \
  --dataset sales.csv
```

---

# Benchmarking

Compare all forecasting models:

```bash
python scripts/benchmark.py
```

---

# Production Features

- Docker Support
- Kubernetes Deployment
- CI/CD Pipelines
- Model Registry
- Drift Detection
- Automated Retraining
- Monitoring Dashboard

---

# Future Roadmap

- Probabilistic Forecasting
- Hierarchical Forecasting
- Global Forecasting Models
- Multimodal Forecasting
- Agentic Forecasting Systems
- Time-Series RAG
- Forecasting Copilot
