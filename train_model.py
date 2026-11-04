import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load corrected monthly sales data
df = pd.read_csv("monthly_sales_forecasting.csv")

# Create numerical month values
df["Month_Number"] = range(len(df))

# Prepare input and output
X = df[["Month_Number"]]
y = df["Sales"]

# Split data: first 9 months for training, last 3 for testing
X_train = X.iloc[:9]
X_test = X.iloc[9:]

y_train = y.iloc[:9]
y_test = y.iloc[9:]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict sales for the test months
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("MODEL EVALUATION RESULTS")
print("------------------------")
print("Training months:", len(X_train))
print("Testing months:", len(X_test))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 3))

# Retrain using all 12 complete months
model.fit(X, y)

# Predict the next month
next_month = pd.DataFrame({
    "Month_Number": [len(df)]
})

prediction = model.predict(next_month)

print("\nNext month's predicted sales:", round(prediction[0], 2))