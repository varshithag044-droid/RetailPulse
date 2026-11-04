import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# File paths
DATA_FILE = Path("retail_prepared.csv")
OUTPUT_FILE = Path("prophet_sales_forecast.csv")

# 1. Load the retail dataset
df = pd.read_csv(DATA_FILE)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

# 2. Clean invalid records
df = df.dropna(subset=["Order_Date", "Quantity", "Unit_Price"])

df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0)
].copy()

# Calculate sales revenue
df["Sales"] = df["Quantity"] * df["Unit_Price"]

# 3. Aggregate sales by day
df["Date"] = df["Order_Date"].dt.floor("D")

daily_sales = (
    df.groupby("Date", as_index=False)["Sales"]
    .sum()
    .rename(columns={
        "Date": "ds",
        "Sales": "y"
    })
)

# 4. Exclude the incomplete final month
daily_sales = daily_sales[
    daily_sales["ds"] < pd.Timestamp("2010-12-01")
].copy()

daily_sales = daily_sales.sort_values("ds").reset_index(drop=True)

if len(daily_sales) < 60:
    raise ValueError(
        "Not enough daily sales data to train and evaluate the model."
    )

# 5. Time-based evaluation
# Keep the last 30 days for testing.
train_data = daily_sales.iloc[:-30].copy()
test_data = daily_sales.iloc[-30:].copy()

# 6. Train Prophet on historical training data
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)

model.fit(train_data)

# 7. Forecast the 30-day test period
test_future = test_data[["ds"]].copy()

test_predictions = model.predict(test_future)

# 8. Evaluate predictions
actual_sales = test_data["y"].to_numpy()
predicted_sales = test_predictions["yhat"].to_numpy()

mae = mean_absolute_error(
    actual_sales,
    predicted_sales
)

# Avoid division by zero when calculating MAPE
nonzero_mask = actual_sales != 0

if nonzero_mask.any():
    mape = (
        abs(
            (
                actual_sales[nonzero_mask]
                - predicted_sales[nonzero_mask]
            )
            / actual_sales[nonzero_mask]
        ).mean()
        * 100
    )
else:
    mape = float("nan")

# 9. Retrain on all available complete months
final_model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95
)

final_model.fit(daily_sales)

# 10. Generate a 30-day future forecast
future_dates = final_model.make_future_dataframe(
    periods=30,
    freq="D"
)

forecast = final_model.predict(future_dates)

# 11. Keep the future predictions
future_forecast = forecast[
    forecast["ds"] > daily_sales["ds"].max()
][
    ["ds", "yhat", "yhat_lower", "yhat_upper"]
].copy()

future_forecast = future_forecast.rename(
    columns={
        "ds": "Date",
        "yhat": "Predicted_Sales",
        "yhat_lower": "Lower_Bound",
        "yhat_upper": "Upper_Bound"
    }
)

# Sales cannot be negative for this demonstration.
for column in [
    "Predicted_Sales",
    "Lower_Bound",
    "Upper_Bound"
]:
    future_forecast[column] = future_forecast[column].clip(lower=0)

# 12. Save the future forecast
future_forecast.to_csv(
    OUTPUT_FILE,
    index=False
)

# 13. Display results
print("\n========== PROPHET FORECAST RESULTS ==========")

print(f"Training days: {len(train_data)}")
print(f"Testing days: {len(test_data)}")

print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

print("\nNext 30-day sales forecast:")
print(future_forecast.head(10).to_string(index=False))

print(f"\nForecast saved to: {OUTPUT_FILE.resolve()}")
print("\nProphet forecasting completed successfully!")