import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# RetailPulse: RFM Customer Segmentation
# ============================================================
# Uses RFM (Recency, Frequency, Monetary) features and K-Means.
# Evaluates k=4..8 and selects the best silhouette score.
# Note: the best-scoring k may be below the project target of
# 6-8 segments. Do not force 6-8 without checking cluster quality.
# ============================================================

INPUT_FILE = "retail_prepared.csv"
OUTPUT_SEGMENTS = "customer_segments.csv"
OUTPUT_SUMMARY = "customer_segment_summary.csv"
OUTPUT_EVALUATION = "cluster_evaluation.csv"


def main():
    # 1. Load and validate data
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "Invoice", "Order_Date", "Quantity", "Unit_Price",
        "Customer_ID", "Sales"
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {INPUT_FILE}: "
            f"{', '.join(sorted(missing_columns))}"
        )

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Customer_ID"] = pd.to_numeric(df["Customer_ID"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    df = df.dropna(
        subset=["Customer_ID", "Order_Date", "Invoice", "Quantity", "Unit_Price"]
    )
    df = df[(df["Quantity"] > 0) & (df["Unit_Price"] > 0)].copy()
    df["Sales"] = df["Quantity"] * df["Unit_Price"]
    if df.empty:
        raise ValueError("No valid purchase rows remain after data cleaning.")

    # 2. Calculate RFM features
    reference_date = df["Order_Date"].max().normalize() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer_ID").agg(
        Last_Purchase=("Order_Date", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Sales", "sum"),
        Total_Units=("Quantity", "sum"),
    ).reset_index()

    rfm["Recency"] = (
        reference_date - rfm["Last_Purchase"].dt.normalize()
    ).dt.days
    rfm = rfm.drop(columns=["Last_Purchase"])
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Recency", "Frequency", "Monetary"]
    )
    rfm = rfm[
        (rfm["Frequency"] >= 1) &
        (rfm["Monetary"] > 0) &
        (rfm["Recency"] >= 0)
    ].copy()

    if len(rfm) < 9:
        raise ValueError("Not enough customers to evaluate 4-8 clusters.")

    print("RFM analysis completed.")
    print(f"Total customers: {len(rfm):,}")
    print(f"Reference date: {reference_date.date()}")

    # 3. Transform skewed features and scale them
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"] = np.log1p(features["Monetary"])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 4. Evaluate candidate cluster counts
    print("\nEvaluating K-Means cluster counts (k=4..8)...")
    evaluation_results = []

    for k in range(4, 9):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(
            scaled_features,
            labels,
            sample_size=min(10000, len(scaled_features)),
            random_state=42,
        )

        counts = pd.Series(labels).value_counts()
        evaluation_results.append({
            "Clusters": k,
            "Silhouette_Score": score,
            "Inertia": model.inertia_,
            "Smallest_Cluster_Size": int(counts.min()),
        })

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_csv(OUTPUT_EVALUATION, index=False)

    print("\nCluster evaluation results:")
    print(evaluation_df.round(4).to_string(index=False))

    best_k = int(
        evaluation_df.loc[evaluation_df["Silhouette_Score"].idxmax(), "Clusters"]
    )
    print(f"\nBest silhouette score is for k={best_k}.")

    # Use the best silhouette result for the primary model.
    # If the project specifically requires 6-8 groups, also produce
    # a k=6 model as a separate target-aligned result for comparison.
    selected_k = best_k
    model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    rfm["Cluster"] = model.fit_predict(scaled_features)

    # 5. Create human-readable segment names based on cluster profiles
    # Use median RFM values to reduce the influence of outliers.
    profile = rfm.groupby("Cluster").agg(
        Customer_Count=("Customer_ID", "size"),
        Median_Recency=("Recency", "median"),
        Median_Frequency=("Frequency", "median"),
        Median_Monetary=("Monetary", "median"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean"),
        Total_Monetary=("Monetary", "sum"),
    )

    recency_median = profile["Median_Recency"].median()
    frequency_median = profile["Median_Frequency"].median()
    monetary_median = profile["Median_Monetary"].median()

    def describe_cluster(row):
        recent = row["Median_Recency"] <= recency_median
        frequent = row["Median_Frequency"] >= frequency_median
        high_value = row["Median_Monetary"] >= monetary_median

        if frequent and high_value and recent:
            return "Recent High-Value Loyal Customers"
        if frequent and high_value:
            return "High-Value Loyal Customers"
        if recent and frequent:
            return "Recent Frequent Customers"
        if recent and high_value:
            return "Recent High-Spending Customers"
        if frequent:
            return "Frequent Customers"
        if high_value:
            return "High-Spending Customers"
        if recent:
            return "Recent Customers"
        return "At-Risk / Low-Activity Customers"

    profile["Customer_Segment"] = profile.apply(describe_cluster, axis=1)

    # Ensure every segment label is unique and retain a stable cluster ID.
    seen = {}
    cluster_names = {}
    for cluster_id in profile.index:
        base_name = profile.loc[cluster_id, "Customer_Segment"]
        seen[base_name] = seen.get(base_name, 0) + 1
        if seen[base_name] == 1:
            cluster_names[cluster_id] = base_name
        else:
            cluster_names[cluster_id] = f"{base_name} (Group {seen[base_name]})"

    rfm["Customer_Segment"] = rfm["Cluster"].map(cluster_names)
    profile["Customer_Segment"] = profile.index.map(cluster_names)

    # 6. Save detailed customer and segment summary outputs
    rfm.to_csv(OUTPUT_SEGMENTS, index=False)

    summary = rfm.groupby(["Cluster", "Customer_Segment"]).agg(
        Customer_Count=("Customer_ID", "nunique"),
        Avg_Recency=("Recency", "mean"),
        Median_Recency=("Recency", "median"),
        Avg_Frequency=("Frequency", "mean"),
        Median_Frequency=("Frequency", "median"),
        Avg_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
        Total_Monetary=("Monetary", "sum"),
    ).reset_index()

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("\nCustomer segmentation completed.")
    print(f"Selected clusters: {selected_k}")
    print("\nCustomers in each segment:")
    print(summary[["Cluster", "Customer_Segment", "Customer_Count"]].to_string(index=False))
    print("\nSegment profile:")
    print(summary.round(2).to_string(index=False))
    print("\nSaved files:")
    print(f"- {OUTPUT_SEGMENTS}")
    print(f"- {OUTPUT_SUMMARY}")
    print(f"- {OUTPUT_EVALUATION}")
    if selected_k < 6:
        print(
            "\nNOTE: The best silhouette model has fewer than 6 clusters. "
            "The project target of 6-8 segments is not yet met. Review the "
            "evaluation results before choosing a target-aligned k."
        )


if __name__ == "__main__":
    main()
