import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("retail_prepared.csv")
OUTPUT_FILE = Path("inventory_recommendations.csv")

# 1. Load retail data
df = pd.read_csv(DATA_FILE)

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Unit_Price"] = pd.to_numeric(
    df["Unit_Price"],
    errors="coerce"
)

# 2. Clean invalid records
df = df.dropna(
    subset=[
        "Order_Date",
        "Product_Code",
        "Product_Name",
        "Quantity",
        "Unit_Price"
    ]
)

df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0)
].copy()

if df.empty:
    raise ValueError("No valid product sales records were found.")

# 3. Aggregate product demand by day
df["Date"] = df["Order_Date"].dt.floor("D")

daily_product_sales = (
    df.groupby(
        ["Product_Code", "Product_Name", "Date"],
        as_index=False
    )["Quantity"]
    .sum()
)

# 4. Calculate the observation period
start_date = daily_product_sales["Date"].min()
end_date = daily_product_sales["Date"].max()

observation_days = (
    end_date.normalize() - start_date.normalize()
).days + 1

if observation_days <= 0:
    raise ValueError("Invalid observation period.")

# 5. Calculate product demand statistics
inventory = (
    daily_product_sales.groupby(
        ["Product_Code", "Product_Name"],
        as_index=False
    )
    .agg(
        Total_Units_Sold=("Quantity", "sum"),
        Selling_Days=("Date", "nunique"),
        Daily_Demand_Std=("Quantity", "std")
    )
)

inventory["Average_Daily_Demand"] = (
    inventory["Total_Units_Sold"] / observation_days
)

inventory["Daily_Demand_Std"] = (
    inventory["Daily_Demand_Std"].fillna(0)
)

# 6. Estimate demand for the next 30 days
inventory["Estimated_30_Day_Demand"] = (
    inventory["Average_Daily_Demand"] * 30
)

# 7. Estimate safety stock
# This is an illustrative buffer based on demand variability.
# Actual safety stock requires service-level and lead-time information.
inventory["Estimated_Safety_Stock"] = (
    1.65
    * inventory["Daily_Demand_Std"]
    * np.sqrt(7)
)

# 8. Calculate a planning target
# Target = estimated 30-day demand + safety-stock buffer.
inventory["Suggested_30_Day_Stock_Target"] = (
    inventory["Estimated_30_Day_Demand"]
    + inventory["Estimated_Safety_Stock"]
)

# 9. Add planning notes
inventory["Planning_Note"] = (
    "Estimate only: current stock and supplier lead time unavailable"
)

# 10. Sort by estimated demand
inventory = inventory.sort_values(
    "Estimated_30_Day_Demand",
    ascending=False
).reset_index(drop=True)

# 11. Round numerical outputs
numeric_columns = [
    "Average_Daily_Demand",
    "Daily_Demand_Std",
    "Estimated_30_Day_Demand",
    "Estimated_Safety_Stock",
    "Suggested_30_Day_Stock_Target"
]

inventory[numeric_columns] = (
    inventory[numeric_columns].round(2)
)

# 12. Save recommendations
inventory.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n========== INVENTORY PLANNING RESULTS ==========")
print(f"Observation period: {start_date.date()} to {end_date.date()}")
print(f"Number of observation days: {observation_days}")
print(f"Products analyzed: {inventory['Product_Code'].nunique():,}")

print("\nTop 10 products by estimated 30-day demand:")

print(
    inventory[
        [
            "Product_Code",
            "Product_Name",
            "Average_Daily_Demand",
            "Estimated_30_Day_Demand",
            "Estimated_Safety_Stock",
            "Suggested_30_Day_Stock_Target"
        ]
    ].head(10).to_string(index=False)
)

print(f"\nSaved to: {OUTPUT_FILE.resolve()}")
print("\nInventory planning completed successfully!")