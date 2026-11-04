import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# File paths
DATA_FILE = Path("retail_prepared.csv")
OUTPUT_FILE = Path("lstm_sales_forecast.csv")

# Configuration
LOOKBACK = 30
FORECAST_DAYS = 30
EPOCHS = 30
BATCH_SIZE = 16

# 1. Load data
df = pd.read_csv(DATA_FILE)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df = df.dropna(
    subset=["Order_Date", "Quantity", "Unit_Price"]
)

# Keep valid positive sales records
df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0)
].copy()

df["Sales"] = df["Quantity"] * df["Unit_Price"]

# 2. Aggregate daily sales
df["Date"] = df["Order_Date"].dt.floor("D")

daily_sales = (
    df.groupby("Date")["Sales"]
    .sum()
    .sort_index()
)

# Exclude incomplete December 2010
daily_sales = daily_sales[
    daily_sales.index < pd.Timestamp("2010-12-01")
]

# Include days without transactions as zero sales
daily_sales = daily_sales.asfreq("D", fill_value=0)

values = daily_sales.to_numpy().reshape(-1, 1)

if len(values) <= LOOKBACK + FORECAST_DAYS:
    raise ValueError("Not enough daily observations for training.")

# 3. Create time-based training and testing periods
train_values = values[:-FORECAST_DAYS]
test_values = values[-FORECAST_DAYS:]

# Fit scaler using training data only
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_values)

# 4. Prepare sequences for LSTM
def create_sequences(data, lookback):
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])
        y.append(data[i, 0])

    return (
        np.array(X).reshape(-1, lookback, 1),
        np.array(y)
    )

X_train, y_train = create_sequences(
    scaled_train,
    LOOKBACK
)

# 5. Build LSTM model
def build_model():
    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model

model = build_model()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# 6. Train the model
model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

# 7. Evaluate the final 30 days recursively
history = values[:-FORECAST_DAYS].flatten().tolist()

predictions = []

for _ in range(FORECAST_DAYS):
    recent = np.array(
        history[-LOOKBACK:]
    ).reshape(-1, 1)

    scaled_recent = scaler.transform(recent)

    X_input = scaled_recent.reshape(
        1, LOOKBACK, 1
    )

    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )[0, 0]

    prediction = scaler.inverse_transform(
        np.array([[prediction_scaled]])
    )[0, 0]

    prediction = max(0, float(prediction))

    predictions.append(prediction)
    history.append(prediction)

actual = test_values.flatten()

mae = mean_absolute_error(
    actual,
    predictions
)

nonzero = actual != 0

if nonzero.any():
    mape = np.mean(
        np.abs(
            (actual[nonzero] - np.array(predictions)[nonzero])
            / actual[nonzero]
        )
    ) * 100
else:
    mape = float("nan")

print("\n========== LSTM EVALUATION ==========")
print(f"Test days: {len(actual)}")
print(f"MAE: {mae:,.2f}")
print(f"MAPE: {mape:.2f}%")

# 8. Retrain on all available data for future forecasting
final_scaler = MinMaxScaler()
scaled_all = final_scaler.fit_transform(values)

X_all, y_all = create_sequences(
    scaled_all,
    LOOKBACK
)

final_model = build_model()

final_model.fit(
    X_all,
    y_all,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    verbose=1
)

# 9. Forecast the next 30 days
history = scaled_all.flatten().tolist()
future_predictions = []

for _ in range(FORECAST_DAYS):
    X_input = np.array(
        history[-LOOKBACK:]
    ).reshape(1, LOOKBACK, 1)

    prediction_scaled = final_model.predict(
        X_input,
        verbose=0
    )[0, 0]

    prediction = final_scaler.inverse_transform(
        np.array([[prediction_scaled]])
    )[0, 0]

    prediction = max(0, float(prediction))

    future_predictions.append(prediction)
    history.append(
        final_scaler.transform(
            np.array([[prediction]])
        )[0, 0]
    )

# 10. Save future forecast
future_dates = pd.date_range(
    start=daily_sales.index.max() + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_predictions
})

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== LSTM FORECAST ==========")
print(forecast_df.head(10).to_string(index=False))
print(f"\nForecast saved to: {OUTPUT_FILE.resolve()}")
print("\nLSTM forecasting completed successfully!")