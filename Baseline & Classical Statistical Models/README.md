# Baseline & Classical Statistical Models

## Overview

Baseline and classical statistical models are the foundation of time-series forecasting.

They are computationally efficient, interpretable, and perform well on univariate datasets with stable patterns.

---

## Implemented Models

### Naive Forecast

Forecast equals the most recent observation.

Best For:

- Benchmarking
- Stable series

---

### Moving Average

Uses the average of previous observations.

Best For:

- Noise reduction
- Short-term forecasting

---

### Autoregressive (AR)

Forecasts future values using previous lagged observations.

Best For:

- Stationary data
- Linear relationships

---

### ARIMA

AutoRegressive Integrated Moving Average.

Components:

- AR
- I
- MA

Best For:

- Trend forecasting
- Financial data
- Economic indicators

---

## Advantages

- Fast
- Explainable
- Low computational cost

## Limitations

- Assumes linearity
- Limited performance on complex patterns