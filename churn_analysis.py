import pandas as pd

# Step 1: Load the prepared retail dataset
df = pd.read_csv("retail_prepared.csv")

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

df = df.dropna(subset=["Customer_ID", "Order_Date"])

# Keep valid purchase transactions
df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0)
].copy()

# Step 2: Define observation and future periods
cutoff_date = pd.Timestamp("2010-09-30")
future_start = pd.Timestamp("2010-10-01")
future_end = pd.Timestamp("2010-11-29")

# Step 3: Use earlier transactions to calculate customer behaviour
observation = df[df["Order_Date"] <= cutoff_date].copy()

customer_features = observation.groupby("Customer_ID").agg(
    Last_Purchase=("Order_Date", "max"),
    Total_Orders=("Invoice", "nunique"),
    Total_Spending=("Sales", "sum"),
    Total_Units=("Quantity", "sum")
).reset_index()

# Calculate days since last purchase at the cutoff date
customer_features["Recency"] = (
    cutoff_date - customer_features["Last_Purchase"]
).dt.days

# NEW: Calculate average order value
customer_features["Average_Order_Value"] = (
    customer_features["Total_Spending"] /
    customer_features["Total_Orders"].clip(lower=1)
)

# NEW: Calculate average units per order
customer_features["Average_Units_Per_Order"] = (
    customer_features["Total_Units"] /
    customer_features["Total_Orders"].clip(lower=1)
)

# Step 4: Check purchases during the following 60 days
future = df[
    (df["Order_Date"] >= future_start) &
    (df["Order_Date"] <= future_end)
]

future_customers = set(future["Customer_ID"].unique())

# Churn label: 1 = no purchase in the future period
customer_features["Churn"] = (
    ~customer_features["Customer_ID"].isin(future_customers)
).astype(int)

# Step 5: Save the dataset
customer_features = customer_features.drop(
    columns=["Last_Purchase"]
)

customer_features.to_csv(
    "churn_model_dataset.csv",
    index=False
)

# Step 6: Display results
print("Churn dataset created successfully!")
print("Total customers:", len(customer_features))

print("\nChurn label counts:")
print(customer_features["Churn"].value_counts())

print("\nChurn percentages:")
print(
    (customer_features["Churn"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nFirst five rows:")
print(customer_features.head())

print("\nSaved file: churn_model_dataset.csv")