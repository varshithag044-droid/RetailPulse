import pandas as pd

# Load the original dataset
df = pd.read_excel("online_retail_II.xlsx")

# Remove duplicate rows
df = df.drop_duplicates()

# Rename columns
df = df.rename(columns={
    "Invoice": "Invoice",
    "StockCode": "Product_Code",
    "Description": "Product_Name",
    "Quantity": "Quantity",
    "InvoiceDate": "Order_Date",
    "Price": "Unit_Price",
    "Customer ID": "Customer_ID",
    "Country": "Country"
})

# Handle missing product names
df["Product_Name"] = df["Product_Name"].fillna("Unknown")

# Convert order date into datetime format
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

# Calculate sales amount
df["Sales"] = df["Quantity"] * df["Unit_Price"]

# Save the prepared dataset
df.to_csv("retail_prepared.csv", index=False)

# Display results
print("Dataset preparation completed!")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five records:")
print(df.head())