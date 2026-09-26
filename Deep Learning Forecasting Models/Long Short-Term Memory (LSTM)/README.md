# Long Short-Term Memory (LSTM)

## Overview

Long Short-Term Memory (LSTM) is a specialized type of Recurrent Neural Network (RNN) designed to learn long-term dependencies in sequential data.

Traditional RNNs suffer from the **vanishing gradient problem**, making it difficult to learn information from long sequences. LSTMs solve this issue using a memory cell and gating mechanisms that control the flow of information.

LSTMs are widely used in:

- Time-Series Forecasting
- Natural Language Processing (NLP)
- Speech Recognition
- Financial Forecasting
- Demand Forecasting
- Sensor Data Analysis

**This tutorial is divided into four parts; they are:**

1. Univariate LSTM Models
    1. Vanilla LSTM
    2. Stacked LSTM
    3. Bidirectional LSTM
    4. CNN LSTM
    5. ConvLSTM
2. Multivariate LSTM Models
    1. Multiple Input Series
    2. Multiple Parallel Series
3. Multi-Step LSTM Models
    1. Vector Output Model
    2. Encoder-Decoder Model
4. Multivariate Multi-Step LSTM Models
    1. Multiple Input Multi-Step Output
    2. Multiple Parallel Input and Multi-Step Output



## Why LSTM?

Traditional neural networks assume inputs are independent of one another.

However, many real-world problems involve sequences:

```text
Yesterday's Sales → Today's Sales → Tomorrow's Sales
```

```text
Previous Stock Prices → Future Stock Price
```

```text
Past Weather Conditions → Future Weather Forecast
```

LSTMs can remember important information over long periods and use it to make better predictions.

A standard RNN struggles to "remember" information from many steps earlier
in a sequence because gradients shrink exponentially as they are
backpropagated through time. LSTM solves this with a **memory cell** and a
system of **gates** that explicitly control what information is kept,
discarded, or output at each time step.

## Common Use Cases

- **Time-series forecasting** — demand, sales, stock prices, sensor data
- **Natural language processing** — language modeling, translation, sentiment analysis (largely superseded by Transformers today)
- **Speech recognition and generation**
- **Anomaly detection** in sequential/streaming data
- **Video and gesture recognition** (as part of larger architectures)

## Strengths and Limitations

**Strengths**
- Captures long-range temporal dependencies better than plain RNNs
- Handles variable-length sequences naturally
- Well-suited to noisy, non-stationary time series with complex patterns
- Mature tooling and broad framework support (PyTorch, TensorFlow/Keras)

**Limitations**
- Sequential computation makes training slower than Transformers, which parallelize across time steps
- Needs more data than classical statistical models (ARIMA, ETS) to perform well
- More hyperparameters to tune (hidden size, layers, window size, learning rate)
- Point forecasts by default — quantifying uncertainty requires extra techniques (e.g. Monte Carlo dropout, quantile loss)
- Largely overtaken by Transformer-based architectures for large-scale NLP, though still competitive for many time-series tasks


## LSTM Architecture

An LSTM cell consists of:

1. Cell State
2. Hidden State
3. Forget Gate
4. Input Gate
5. Output Gate

```text
                ┌─────────────┐
                │ Cell State  │
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Forget Gate    Input Gate    Output Gate
        │              │              │
        └──────┬───────┴───────┬──────┘
               ▼               ▼
            LSTM Memory Cell
               │
               ▼
          Hidden State
```


## Core Components

### Cell State

The cell state acts as the long-term memory of the network.

It carries relevant information through the sequence with minimal modifications.

$$
C_t
$$

Where:

- $C_t$ = Current Cell State

### Hidden State

The hidden state represents the short-term memory and output of the LSTM at each time step.

$$
h_t
$$

Where:

- $h_t$ = Hidden State

## Forget Gate

The forget gate decides what information should be removed from memory.

$$
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
$$

Where:

| Symbol | Description |
|----------|-------------|
| $f_t$ | Forget gate output |
| $W_f$ | Forget gate weights |
| $h_{t-1}$ | Previous hidden state |
| $x_t$ | Current input |
| $b_f$ | Bias |
| $\sigma$ | Sigmoid activation |

Output range:

$$
0 \le f_t \le 1
$$

- 0 → Forget everything
- 1 → Keep everything

## Input Gate

The input gate determines what new information should be stored.

### Step 1: Gate Decision

$$
i_t = \sigma(W_i[h_{t-1},x_t] + b_i)
$$

### Step 2: Candidate Memory

$$
\tilde{C_t} = tanh(W_c[h_{t-1},x_t] + b_c)
$$

### Step 3: Update Cell State

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C_t}
$$

Where:

- $\odot$ = Element-wise multiplication

---

## Output Gate

The output gate determines what information is exposed as the hidden state.

$$
o_t = \sigma(W_o[h_{t-1},x_t] + b_o)
$$

Updated hidden state:

$$
h_t = o_t \odot tanh(C_t)
$$


## Complete LSTM Flow

```text
Input
  │
  ▼
Forget Gate
  │
  ▼
Input Gate
  │
  ▼
Update Memory Cell
  │
  ▼
Output Gate
  │
  ▼
Hidden State
```

---

## LSTM for Time-Series Forecasting

Example:

```text
Day 1: 100
Day 2: 105
Day 3: 110
Day 4: 112
Day 5: ?
```

Input Sequence:

```text
[100, 105, 110, 112]
```

Output:

```text
115
```

The LSTM learns temporal dependencies and predicts the next value.

---

## Data Preparation

LSTM expects input in a 3D format:

```python
(samples, time_steps, features)
```

Example:

```python
(1000, 30, 5)
```

Meaning:

| Dimension | Description |
|------------|-------------|
| 1000 | Samples |
| 30 | Time Steps |
| 5 | Features |

---

# Building an LSTM Model

## TensorFlow / Keras Example

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(
        units=64,
        input_shape=(30, 1)
    ),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()
```

---

# Multi-Layer LSTM

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(
        128,
        return_sequences=True,
        input_shape=(30,1)
    ),
    LSTM(64),
    Dense(1)
])
```

---

# Hyperparameters

## Units

Number of memory cells.

Common values:

```text
32
64
128
256
```

### Sequence Length

```text
30 Days
60 Days
90 Days
```

### Batch Size

```text
16
32
64
128
```

### Learning Rate

```text
0.001
0.0005
0.0001
```

### Epochs

```text
50 – 500
```

---

# Evaluation Metrics

## Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y_i}|
$$

## Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2}
$$

## Mean Absolute Percentage Error (MAPE)

$$
MAPE = \frac{100}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat{y_i}}{y_i}
\right|
$$

---

# Advantages

✅ Captures long-term dependencies

✅ Handles nonlinear patterns

✅ Learns temporal relationships automatically

✅ Effective for sequential data

✅ Strong forecasting performance

---

# Limitations

❌ Requires large datasets

❌ Computationally expensive

❌ Slower training

❌ Difficult to interpret

❌ Risk of overfitting

---

# LSTM vs Other Models

| Model | Long-Term Memory | Nonlinear Relationships | Large Datasets |
|---------|-----------------|------------------------|---------------|
| ARIMA | ❌ | ❌ | ❌ |
| XGBoost | ❌ | ✅ | ✅ |
| LSTM | ✅ | ✅ | ✅ |
| GRU | ✅ | ✅ | ✅ |
| Transformer | ✅ | ✅ | ✅ |

---

# Use Cases

## Finance

- Stock Price Forecasting
- Cryptocurrency Forecasting
- Volatility Prediction

## Retail

- Demand Forecasting
- Revenue Forecasting
- Inventory Planning

## Energy

- Load Forecasting
- Power Consumption Prediction

## Manufacturing

- Predictive Maintenance
- Sensor Forecasting

## Weather

- Temperature Forecasting
- Rainfall Prediction

---

# Best Practices

- Normalize data before training.
- Use sliding windows for sequence generation.
- Apply dropout regularization.
- Monitor validation loss.
- Use walk-forward validation.
- Compare against baseline models.

---

# References

1. Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
2. Deep Learning — Ian Goodfellow
3. TensorFlow Documentation
4. PyTorch Documentation
5. Forecasting: Principles and Practice
