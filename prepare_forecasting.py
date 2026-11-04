import pandas as pd

# Load the prepared retail dataset
df = pd.read_csv("retail_prepared.csv")

# Convert Order_Date into datetime format
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

# Keep valid sales records
df = df.dropna(subset=["Order_Date"])

df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0)
].copy()

# Calculate sales revenue
df["Sales"] = df["Quantity"] * df["Unit_Price"]

# Group sales by month
df["Month"] = df["Order_Date"].dt.to_period("M")

monthly_sales = (
    df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

# Exclude incomplete December 2010
monthly_sales = monthly_sales[
    monthly_sales["Month"] < pd.Period("2010-12", freq="M")
].copy()

# Convert month into a date
monthly_sales["Month"] = (
    monthly_sales["Month"].dt.to_timestamp()
)

# Save the monthly sales data
monthly_sales.to_csv(
    "monthly_sales_forecasting.csv",
    index=False
)

print("Monthly sales data prepared successfully!")
print("\nFirst 10 rows:")
print(monthly_sales.head(10))

print("\nTotal months:", len(monthly_sales))