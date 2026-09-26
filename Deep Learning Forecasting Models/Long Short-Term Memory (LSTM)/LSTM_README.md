# Long Short-Term Memory (LSTM)

A guide to what LSTM networks are, how they work, and how to use the
`LSTMForecaster` from this project's forecasting toolkit.

## What is LSTM?

Long Short-Term Memory (LSTM) is a type of Recurrent Neural Network (RNN)
architecture designed to learn patterns in sequential data — text, audio,
sensor readings, or time series — while avoiding the **vanishing gradient
problem** that limits plain RNNs from learning long-range dependencies.

LSTMs were introduced by Sepp Hochreiter and Jürgen Schmidhuber in 1997 and
remain one of the most widely used architectures for sequence modeling,
particularly before the rise of Transformers.

## Why LSTM Instead of a Plain RNN?

A standard RNN struggles to "remember" information from many steps earlier
in a sequence because gradients shrink exponentially as they are
backpropagated through time. LSTM solves this with a **memory cell** and a
system of **gates** that explicitly control what information is kept,
discarded, or output at each time step.

## Architecture

Each LSTM unit (cell) maintains two states passed from one time step to the
next:

- **Cell state (`C_t`)** — a long-term memory that flows through the
  sequence with only minor, gated modifications.
- **Hidden state (`h_t`)** — the short-term output used for predictions and
  passed to the next layer/step.

Three gates control the flow of information at every time step `t`:

| Gate | Purpose | Formula |
|---|---|---|
| **Forget gate** | Decides what to discard from the cell state | `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)` |
| **Input gate** | Decides what new information to store | `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)` |
| **Output gate** | Decides what to output as the hidden state | `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)` |

Candidate values and state updates:

```
C̃_t  = tanh(W_C · [h_{t-1}, x_t] + b_C)      # candidate cell state
C_t  = f_t * C_{t-1} + i_t * C̃_t             # new cell state
h_t  = o_t * tanh(C_t)                        # new hidden state
```

Where `σ` is the sigmoid function (outputs 0–1, acting as a "gate"), `tanh`
squashes values to (-1, 1), and `*` is element-wise multiplication.

```
                 ┌─────────────────────────────────────┐
                 │              LSTM Cell               │
                 │                                       │
  C_{t-1} ───────┼──(×)───────────────(+)────────────────┼─── C_t
                 │   │  forget         │  ▲               │
                 │   │  gate            \ | input          │
                 │   ▼                  \|  gate           │
  h_{t-1} ──┬────┼─[σ]      [σ]────(×)   [tanh]            │
       x_t ─┘    │           candidate                      │
                 │                                       │
  h_{t-1} ──┬────┼─────────[σ]──────(×)──[tanh]──────────┼─── h_t
       x_t ─┘    │        output gate                     │
                 └─────────────────────────────────────┘
```

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

## Using `LSTMForecaster` in This Toolkit

The toolkit wraps a PyTorch LSTM behind the same `BaseForecaster` interface
used by every other model (ARIMA, XGBoost, Prophet, etc.), so it's a
drop-in replacement in any pipeline.

### Installation

```bash
pip install -r requirements.txt
pip install torch          # LSTM/GRU/Transformer models need PyTorch
```

### Basic usage

```python
from tsforecast.data.loader import generate_synthetic_series
from tsforecast.preprocessing.transforms import train_test_split_ts
from tsforecast.models.deep_learning import LSTMForecaster
from tsforecast.evaluation.metrics import evaluate_all

series = generate_synthetic_series(n=365, freq="D", seasonality_period=7)
train, test = train_test_split_ts(series, test_size=30)

model = LSTMForecaster(
    window_size=30,     # how many past steps the model looks at
    hidden_size=32,     # size of the LSTM's hidden state
    num_layers=1,       # number of stacked LSTM layers
    epochs=50,
    lr=1e-3,
    batch_size=16,
)
model.fit(train)
forecast = model.predict(horizon=len(test))

print(evaluate_all(test.values, forecast, y_train=train.values, seasonal_period=7))
```

### Via the CLI

```bash
python scripts/train.py --model lstm --data data/raw/sample.csv \
    --target value --date-col date --horizon 30 --epochs 50
```

### How forecasting works internally

1. The series is standardized (`StandardScalerTS`) so the network trains on
   a zero-mean, unit-variance signal.
2. A sliding window of the last `window_size` points is used as input to
   predict the next single point (`make_windowed_dataset`).
3. The LSTM is trained with MSE loss via the Adam optimizer.
4. At inference time, forecasting is **recursive**: the model predicts one
   step ahead, appends that prediction to its input window, and repeats
   until it has produced `horizon` steps.

### Key hyperparameters to tune

| Parameter | Effect |
|---|---|
| `window_size` | How much history informs each prediction. Too short misses long-range patterns; too long slows training and can overfit. |
| `hidden_size` | Model capacity. Larger values fit more complex patterns but risk overfitting on small datasets. |
| `num_layers` | Depth of the stacked LSTM. Rarely needs to exceed 2–3 for typical forecasting tasks. |
| `epochs` / `lr` | Standard training controls — watch for over/underfitting. |

### When to reach for LSTM vs. other models in this toolkit

- Prefer **ARIMA/SARIMA/ETS** for short series with clear, stable seasonality — they train instantly and are easy to interpret.
- Prefer **XGBoost/LightGBM/RandomForest** when you have useful external features (holidays, promotions, weather) alongside the series.
- Prefer **LSTM/GRU** when the series is long, the patterns are complex or non-linear, and you have enough data to train a neural network.
- Prefer **Transformer** over LSTM when the sequence is very long and you can afford more compute — attention scales better with long-range dependencies.
- Prefer **Prophet** when you need fast, interpretable trend/seasonality/holiday decomposition with minimal tuning.

## References

- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- Olah, C. (2015). [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — a widely cited visual explanation.
- PyTorch documentation: [`torch.nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
