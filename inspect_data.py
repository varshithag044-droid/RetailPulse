import pandas as pd

df = pd.read_excel("online_retail_ii.xlsx")

print(df.head())
print("Dataset shape:")
print(df.shape)
print("Column names:")
print(df.columns.tolist())
print("Missing values in each column:")
print(df.isnull().sum())

print("Duplicate rows:")
print(df.duplicated().sum())