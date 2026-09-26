# ARIMA (AutoRegressive Integrated Moving Average)

## Overview

ARIMA (AutoRegressive Integrated Moving Average) is one of the most widely used statistical models for time-series forecasting. It combines three components:

1. **Autoregression (AR)** – Uses past observations to predict future values.
2. **Integration (I)** – Applies differencing to make the series stationary.
3. **Moving Average (MA)** – Uses past forecast errors to improve predictions.

ARIMA is particularly effective for univariate time-series data exhibiting trends but no strong seasonal patterns.

---

# Mathematical Representation

An ARIMA model is denoted as:

$$
ARIMA(p,d,q)
$$

Where:

| Parameter | Description |
|------------|-------------|
| p | Number of autoregressive (AR) terms |
| d | Number of differencing operations |
| q | Number of moving average (MA) terms |

---

# ARIMA Components

## 1. Autoregressive (AR)

### AR(p) Model

The Autoregressive (AR) model predicts the current value of a time series using its previous values.

$$
Y_t = c + \sum_{i=1}^{p}\phi_i Y_{t-i} + \epsilon_t
$$

Where:

| Symbol | Description |
|---------|-------------|
| $Y_t$ | Current value of the time series |
| $Y_{t-i}$ | Previous lagged observations |
| $c$ | Constant term (intercept) |
| $\phi_i$ | Autoregressive coefficients |
| $\epsilon_t$ | Random error term (white noise) |
| $p$ | Number of lag observations used |

#### Example: AR(2)

For an AR model with two lag terms:

$$
Y_t = c + \phi_1Y_{t-1} + \phi_2Y_{t-2} + \epsilon_t
$$

This means the current value depends on:

- Previous value: $Y_{t-1}$
- Second previous value: $Y_{t-2}$
- Random noise: $\epsilon_t$

#### Use Cases

- Stock price forecasting
- Sales forecasting
- Demand prediction
- Economic indicator forecasting

#### Advantages

- Simple and interpretable
- Captures temporal dependencies
- Works well for stationary time series

#### Limitations

- Assumes linear relationships
- Requires stationary data
- Performance decreases for highly nonlinear patterns

---

## 2. Integrated (I)

Differencing removes trends and makes the series stationary.

### First Order Differencing

First-order differencing removes linear trends by subtracting the previous observation from the current observation.

$$
Y'_t = Y_t - Y_{t-1}
$$

Where:

- $Y_t$ = Current value
- $Y_{t-1}$ = Previous value
- $Y'_t$ = Differenced value

Example:

| Time | Value |
|------|--------|
| t-1 | 100 |
| t | 110 |

$$
Y'_t = 110 - 100 = 10
$$

---

### Second Order Differencing

If the first differencing does not make the series stationary, a second differencing can be applied.

$$
Y''_t = Y'_t - Y'_{t-1}
$$

Expanding the equation:

$$
Y''_t = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2})
$$

Simplified:

$$
Y''_t = Y_t - 2Y_{t-1} + Y_{t-2}
$$

Where:

- $Y''_t$ = Second-order differenced value
- $Y'_t$ = First-order differenced value

Example:

| Time | Value |
|------|--------|
| t-2 | 90 |
| t-1 | 100 |
| t | 110 |

First Difference:

$$
Y'_t = 110 - 100 = 10
$$

$$
Y'_{t-1} = 100 - 90 = 10
$$

Second Difference:

$$
Y''_t = 10 - 10 = 0
$$

A second difference of zero indicates that the underlying trend has been removed.

### Why Differencing?

Many statistical forecasting methods assume stationarity:

- Constant mean
- Constant variance
- Constant autocorrelation

---

## 3. Moving Average (MA)

Uses previous forecast errors to improve future predictions.

### MA(q) Model

The Moving Average (MA) model predicts the current value of a time series using previous forecast errors.

$$
Y_t = \mu + \epsilon_t + \sum_{i=1}^{q}\theta_i\epsilon_{t-i}
$$

Where:

| Symbol | Description |
|---------|-------------|
| $Y_t$ | Current value of the time series |
| $\mu$ | Mean of the series |
| $\epsilon_t$ | Current random error (white noise) |
| $\epsilon_{t-i}$ | Previous forecast errors |
| $\theta_i$ | Moving Average coefficients |
| $q$ | Number of lagged error terms |

---

### Example: MA(2)

For a Moving Average model of order 2:

$$
Y_t = \mu + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2\epsilon_{t-2}
$$

This means the current value depends on:

- Current random error: $\epsilon_t$
- Previous error: $\epsilon_{t-1}$
- Second previous error: $\epsilon_{t-2}$

---

### Interpretation

Suppose:

$$
\mu = 100
$$

$$
\epsilon_t = 2
$$

$$
\epsilon_{t-1} = -3
$$

$$
\epsilon_{t-2} = 1
$$

$$
\theta_1 = 0.5
$$

$$
\theta_2 = 0.3
$$

Then:

$$
Y_t = 100 + 2 + (0.5 \times -3) + (0.3 \times 1)
$$

$$
Y_t = 100 + 2 - 1.5 + 0.3
$$

$$
Y_t = 100.8
$$

---

### When to Use MA Models

MA models are useful when:

- Forecast errors exhibit autocorrelation
- Recent shocks affect future observations
- Data is stationary
- Short-term forecasting is required

---

### Advantages

✅ Captures the impact of past forecast errors

✅ Simple statistical interpretation

✅ Effective for stationary series

✅ Useful for modeling short-term fluctuations

---

### Limitations

❌ Requires stationary data

❌ Cannot model trends directly

❌ Cannot model seasonality directly

❌ Performance decreases for highly nonlinear patterns

---

# Complete ARIMA Model

ARIMA combines:

- **AR (AutoRegressive)** – Uses previous observations.
- **I (Integrated)** – Applies differencing to achieve stationarity.
- **MA (Moving Average)** – Uses previous forecast errors.

The complete model is represented as:

$$
ARIMA(p,d,q)
$$

Where:

| Parameter | Description |
|-----------|-------------|
| $p$ | Number of Autoregressive (AR) terms |
| $d$ | Number of Differencing operations |
| $q$ | Number of Moving Average (MA) terms |


### Example

$$
ARIMA(2,1,1)
$$

Meaning:

| Component | Value | Description |
|------------|--------|-------------|
| AR Order ($p$) | 2 | Uses the previous 2 observations |
| Differencing ($d$) | 1 | Applies first-order differencing |
| MA Order ($q$) | 1 | Uses the previous forecast error |

### Interpretation

An ARIMA(2,1,1) model:

1. Applies first-order differencing to make the series stationary.
2. Uses the previous two observations to predict future values.
3. Uses the previous forecast error to improve prediction accuracy.

# Workflow for Building ARIMA Models

```text
Raw Time Series
      │
      ▼
Check Stationarity
      │
      ▼
Apply Differencing 
      │
      ▼
Determine p and q
      │
      ▼
Train ARIMA Model
      │
      ▼
Evaluate Forecast
      │
      ▼
Deploy Model
```

---

# Step 1: Check Stationarity

ARIMA requires stationary data.

Common methods:

### Visual Inspection

- Trend analysis
- Rolling mean
- Rolling variance

### Statistical Tests

#### Augmented Dickey-Fuller (ADF) Test

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
```

Interpretation:

| p-value | Result |
|----------|---------|
| < 0.05 | Stationary |
| > 0.05 | Non-stationary |

---

# Step 2: Determine AR and MA Orders

## ACF (Autocorrelation Function)

Used to determine:

```text
q (MA order)
```

## PACF (Partial Autocorrelation Function)

Used to determine:

```text
p (AR order)
```

### Example

```python
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
```

---

# Step 3: Train ARIMA Model

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(
    series,
    order=(2,1,1)
)

model_fit = model.fit()
```

---

# Step 4: Forecast

```python
forecast = model_fit.forecast(
    steps=30
)

print(forecast)
```

---

# Example

## Monthly Sales Forecasting

Dataset:

```text
Month     Sales
Jan       120
Feb       132
Mar       145
...
```

Model:

```text
ARIMA(1,1,1)
```

Output:

```text
Next Month Forecast:
152 Units
```

---

# Hyperparameters

## p – Autoregressive Order

Controls how many past observations are used.

Example:

```text
p = 3
```

Uses:

```text
Y(t-1), Y(t-2), Y(t-3)
```

---

## d – Differencing Order

Controls stationarity.

Typical values:

```text
0, 1, 2
```

Most real-world datasets:

```text
d = 1
```

---

## q – Moving Average Order

Controls how many previous errors are considered.

Example:

```text
q = 2
```

Uses:

```text
Error(t-1)
Error(t-2)
```

---

# Model Selection

## Grid Search

```python
for p in range(5):
    for d in range(3):
        for q in range(5):
            ...
```

---

## Auto ARIMA

Automatically finds the best parameters.

```python
from pmdarima import auto_arima

model = auto_arima(
    series,
    seasonal=False
)
```

---

# Evaluation Metrics

Common forecasting metrics:

## MAE

Mean Absolute Error

\[
MAE = \frac{1}{n}\sum|y-\hat y|
\]

---

## RMSE

Root Mean Squared Error

\[
RMSE=\sqrt{\frac{1}{n}\sum(y-\hat y)^2}
\]

---

## MAPE

Mean Absolute Percentage Error

\[
MAPE=\frac{100}{n}\sum\left|\frac{y-\hat y}{y}\right|
\]

---

# Advantages

✅ Simple and interpretable

✅ Strong statistical foundation

✅ Works well on small datasets

✅ Effective for linear relationships

✅ Good baseline forecasting model

✅ Fast training

---

# Limitations

❌ Assumes linear relationships

❌ Cannot capture nonlinear patterns

❌ Requires stationary data

❌ Poor performance on highly complex datasets

❌ Limited handling of multiple seasonalities

---

# ARIMA vs Other Models

| Model | Trend | Seasonality | Nonlinear Data |
|---------|--------|-------------|----------------|
| ARIMA | ✅ | ❌ | ❌ |
| SARIMA | ✅ | ✅ | ❌ |
| Prophet | ✅ | ✅ | Limited |
| XGBoost | ✅ | ✅ | ✅ |
| LSTM | ✅ | ✅ | ✅ |
| Transformer | ✅ | ✅ | ✅ |

---

# Use Cases

## Finance

- Stock Prices
- Exchange Rates
- Commodity Prices

## Business

- Revenue Forecasting
- Sales Forecasting
- Demand Planning

## Economics

- GDP Forecasting
- Inflation Forecasting
- Employment Analysis

## Operations

- Inventory Planning
- Capacity Forecasting
- Resource Allocation

---

# Best Practices

- Always check stationarity before training.
- Use differencing carefully to avoid over-differencing.
- Analyze ACF and PACF plots before selecting parameters.
- Compare against simpler baseline models.
- Validate forecasts using walk-forward validation.
- Consider SARIMA if seasonality exists.

---

# Directory Structure

```text
src/models/baseline/
│
├── README.md
├── arima.py
├── auto_arima.py
├── arima_forecaster.py
└── arima_utils.py
```

---

# References

- Box, Jenkins & Reinsel – Time Series Analysis: Forecasting and Control
- Hyndman & Athanasopoulos – Forecasting: Principles and Practice
- Statsmodels Documentation
- pmdarima Documentation
- Introduction to Time Series and Forecasting – Brockwell & Davis
