import pandas as pd
import matplotlib.pyplot as plt

# Load the prepared dataset
df = pd.read_csv("retail_prepared.csv")

# Convert date to datetime
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

# Keep valid sales records
df = df.dropna(subset=["Order_Date"])

df = df[
    (df["Quantity"] > 0) &
    (df["Unit_Price"] > 0)
].copy()

# Calculate sales
df["Sales"] = df["Quantity"] * df["Unit_Price"]

# Extract month
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# Calculate monthly sales
monthly_sales = df.groupby("Month")["Sales"].sum()

# Create the chart
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind="line", marker="o")

plt.title("RetailPulse - Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Revenue")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Save the chart
plt.savefig("monthly_sales.png", dpi=150)

plt.show()

print("Monthly sales chart created successfully!")
# Top 10 products by quantity sold
top_products = (
    df.groupby("Product_Name")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Create bar chart
plt.figure(figsize=(12, 6))

top_products.sort_values().plot(
    kind="barh"
)

plt.title("RetailPulse - Top 10 Best-Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product Name")
plt.tight_layout()

# Save chart
plt.savefig("top_products.png", dpi=150)

plt.show()

print("Top products chart created successfully!")
# Top 10 countries by sales
country_sales = (
    df.groupby("Country")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# Create horizontal bar chart
plt.figure(figsize=(12, 6))

country_sales.sort_values().plot(
    kind="barh"
)

plt.title("RetailPulse - Top 10 Countries by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Country")
plt.tight_layout()

# Save chart
plt.savefig("country_sales.png", dpi=150)

plt.show()

print("Country-wise sales chart created successfully!")