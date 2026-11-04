import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetailPulse Dashboard",
    page_icon="📊",
    layout="wide"
)
# ---------------------------------------------------------
# RetailPulse Professional Dashboard Styling
# ---------------------------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0E1117;
}

/* Main headings */
h1, h2, h3 {
    color: #F8FAFC;
    font-weight: 700;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #171C28;
}

/* Sidebar text */
[data-testid="stSidebar"] p {
    color: #D1D5DB;
}

/* KPI metric cards */
[data-testid="stMetric"] {
    background-color: #1B2333;
    border: 1px solid #303B50;
    padding: 20px;
    border-radius: 12px;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    color: #AAB7CC;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #60A5FA;
    font-weight: bold;
}

/* Buttons */
.stDownloadButton button {
    border-radius: 8px;
    border: 1px solid #3B82F6;
    background-color: #1D4ED8;
    color: white;
    font-weight: 600;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #303B50;
    border-radius: 10px;
}

/* Section separators */
hr {
    border-color: #303B50;
}

</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------
# Load and prepare retail data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("retail_prepared.csv")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Unit_Price"] = pd.to_numeric(df["Unit_Price"], errors="coerce")
    df["Sales"] = df["Quantity"] * df["Unit_Price"]
    return df

df = load_data()

st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
st.title("📊 RetailPulse | Retail Intelligence Dashboard")
st.markdown(
    "<p style='color:#94A3B8; font-size:18px;'>"
    "AI-Powered Customer Analytics, Sales Intelligence & Demand Forecasting"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")


# ---------------------------------------------------------
# Sidebar quick navigation
# ---------------------------------------------------------
st.sidebar.title("📍 RetailPulse Index")
st.sidebar.caption("Click a section to jump directly to it")
st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] a { text-decoration: none; }
    [data-testid="stSidebar"] a:hover { text-decoration: underline; }
    </style>
    <div style="display:flex;flex-direction:column;gap:10px;font-size:15px;">
      <a href="#overview">🏠 Overview</a>
      <a href="#sales-analytics">📈 Sales Analytics</a>
      <a href="#product-performance">🏆 Product Performance</a>
      <a href="#customer-analytics">👥 Customer Analytics</a>
      <a href="#monthly-forecast">🤖 Monthly Forecast</a>
      <a href="#prophet-forecast">🔮 Prophet Forecast</a>
      <a href="#customer-segmentation">🧩 Customer Segmentation</a>
      <a href="#inventory-optimization">📦 Inventory Optimization</a>
      <a href="#what-if-analysis">🎛️ What-If Analysis</a>
      <a href="#lstm-forecast">🧠 LSTM Forecast</a>
      <a href="#model-comparison">📊 Model Comparison</a>
      <a href="#churn-prediction">⚠️ Churn Prediction</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# Valid positive sales records
valid_df = df.dropna(subset=["Order_Date"]).copy()
valid_df = valid_df[
    (valid_df["Quantity"] > 0) &
    (valid_df["Unit_Price"] > 0)
].copy()

# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------
total_sales = valid_df["Sales"].sum()
total_orders = valid_df["Invoice"].nunique()
total_customers = valid_df["Customer_ID"].nunique()
total_units = valid_df["Quantity"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales Revenue", f"£{total_sales / 1_000_000:.2f}M")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Total Customers", f"{total_customers:,}")
col4.metric("Total Units Sold", f"{total_units:,.0f}")

st.markdown("---")

# ---------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
st.success("RetailPulse dashboard is running!")

# ---------------------------------------------------------
# Country filter
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="sales-analytics"></div>', unsafe_allow_html=True)
st.subheader("🔎 Filter Sales by Country")
countries = sorted(valid_df["Country"].dropna().unique().tolist())
selected_country = st.selectbox(
    "Select a Country",
    ["All Countries"] + countries
)

if selected_country == "All Countries":
    filtered_df = valid_df.copy()
else:
    filtered_df = valid_df[valid_df["Country"] == selected_country].copy()

st.metric("Sales for Selected Country", f"£{filtered_df['Sales'].sum():,.2f}")
st.dataframe(filtered_df.head(10), use_container_width=True)

# ---------------------------------------------------------
# Monthly sales trend
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Monthly Sales Trend")
monthly_df = filtered_df.copy()
monthly_df["Month"] = monthly_df["Order_Date"].dt.to_period("M").astype(str)
monthly_sales = (
    monthly_df.groupby("Month", as_index=False)["Sales"]
    .sum()
)
st.line_chart(
    monthly_sales,
    x="Month",
    y="Sales",
    color="#38BDF8",
    use_container_width=True
)

# ---------------------------------------------------------
# Best-selling products
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="product-performance"></div>', unsafe_allow_html=True)
st.subheader("🏆 Top 10 Best-Selling Products")
product_sales = (
    filtered_df.groupby("Product_Name")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
st.bar_chart(
    product_sales,
    x="Product_Name",
    y="Quantity",
    horizontal=True,
    color="#38BDF8",
    use_container_width=True
)

# ---------------------------------------------------------
# Customer analytics
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="customer-analytics"></div>', unsafe_allow_html=True)
st.subheader("👥 Top 10 Customers by Spending")
customer_sales = (
    filtered_df.dropna(subset=["Customer_ID"])
    .groupby("Customer_ID")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
customer_sales.columns = ["Customer ID", "Total Spending"]
st.bar_chart(
    customer_sales,
    x="Customer ID",
    y="Total Spending",
    horizontal=True
)

# ---------------------------------------------------------
# Monthly linear-regression forecast (existing feature)
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="monthly-forecast"></div>', unsafe_allow_html=True)
st.subheader("🤖 Monthly Sales Forecast")
try:
    forecast_df = pd.read_csv("monthly_sales_forecasting.csv")
    if "Sales" not in forecast_df.columns:
        st.warning("monthly_sales_forecasting.csv must contain a 'Sales' column.")
    elif len(forecast_df) < 2:
        st.warning("At least two months of data are needed for the monthly forecast.")
    else:
        forecast_df["Sales"] = pd.to_numeric(forecast_df["Sales"], errors="coerce")
        forecast_df = forecast_df.dropna(subset=["Sales"]).reset_index(drop=True)
        forecast_df["Month_Number"] = range(len(forecast_df))
        X = forecast_df[["Month_Number"]]
        y = forecast_df["Sales"]
        model = LinearRegression()
        model.fit(X, y)
        next_month = pd.DataFrame({"Month_Number": [len(forecast_df)]})
        predicted_sales = max(0, float(model.predict(next_month)[0]))
        st.metric("Estimated Sales for Next Month", f"£{predicted_sales:,.2f}")
        st.info(
            "This is a preliminary linear-regression estimate based on historical monthly sales. "
            "Validate the forecast before using it for business decisions."
        )
except FileNotFoundError:
    st.warning("Monthly forecast file not found: monthly_sales_forecasting.csv")

# ---------------------------------------------------------
# Prophet 30-day forecast
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="prophet-forecast"></div>', unsafe_allow_html=True)
st.subheader("🔮 Prophet: 30-Day Sales Forecast")
try:
    prophet_df = pd.read_csv("prophet_sales_forecast.csv")
    required_prophet = {"Date", "Predicted_Sales", "Lower_Bound", "Upper_Bound"}
    if not required_prophet.issubset(prophet_df.columns):
        st.warning("The Prophet forecast CSV is missing one or more required columns.")
    else:
        prophet_df["Date"] = pd.to_datetime(prophet_df["Date"], errors="coerce")
        prophet_df["Predicted_Sales"] = pd.to_numeric(prophet_df["Predicted_Sales"], errors="coerce")
        prophet_df["Lower_Bound"] = pd.to_numeric(prophet_df["Lower_Bound"], errors="coerce")
        prophet_df["Upper_Bound"] = pd.to_numeric(prophet_df["Upper_Bound"], errors="coerce")
        prophet_df = prophet_df.dropna(subset=["Date", "Predicted_Sales"])
        st.write("This forecast uses the Prophet model trained on historical daily sales data.")
        st.write("### Daily Predicted Sales")
        st.line_chart(prophet_df, x="Date", y="Predicted_Sales")
        st.write("### Forecast Bounds")
        bounds_df = prophet_df.set_index("Date")[["Lower_Bound", "Upper_Bound"]]
        st.line_chart(bounds_df)
        col1, col2 = st.columns(2)
        col1.metric("Average Predicted Daily Sales", f"£{prophet_df['Predicted_Sales'].mean():,.2f}")
        col2.metric("Total Predicted Sales (30 Days)", f"£{prophet_df['Predicted_Sales'].sum():,.2f}")
        st.download_button(
            "📥 Download Prophet Forecast CSV",
            data=prophet_df.to_csv(index=False).encode("utf-8"),
            file_name="prophet_sales_forecast.csv",
            mime="text/csv",
            key="download_prophet_forecast"
        )
except FileNotFoundError:
    st.warning("Prophet forecast file not found. Run prophet_forecasting.py first.")

# ---------------------------------------------------------
# Customer segmentation
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="customer-segmentation"></div>', unsafe_allow_html=True)
st.subheader("👥 Customer Segmentation Analysis")
try:
    segments_df = pd.read_csv("customer_segments.csv")
    required_segment_cols = {
        "Customer_ID", "Recency", "Frequency", "Monetary", "Customer_Segment"
    }
    if not required_segment_cols.issubset(segments_df.columns):
        st.warning("customer_segments.csv is missing required columns.")
    else:
        st.write("### Customer Groups")
        segment_counts = (
            segments_df["Customer_Segment"]
            .value_counts()
            .rename_axis("Customer Segment")
            .reset_index(name="Number of Customers")
        )
        st.bar_chart(
            segment_counts,
            x="Customer Segment",
            y="Number of Customers",
            horizontal=True
        )
        st.write("### Explore Customer Groups")
        segment_options = ["All Segments"] + sorted(
            segments_df["Customer_Segment"].dropna().unique().tolist()
        )
        selected_segment = st.selectbox(
            "Select a Customer Segment",
            segment_options
        )
        if selected_segment == "All Segments":
            selected_customers = segments_df.copy()
        else:
            selected_customers = segments_df[
                segments_df["Customer_Segment"] == selected_segment
            ].copy()
        st.metric("Customers in Selected Group", f"{len(selected_customers):,}")
        st.dataframe(
            selected_customers[
                ["Customer_ID", "Recency", "Frequency", "Monetary", "Customer_Segment"]
            ],
            use_container_width=True,
            hide_index=True
        )
except FileNotFoundError:
    st.warning("Customer segments file not found. Run customer_segmentation.py first.")

# ---------------------------------------------------------
# Inventory optimization / demand planning
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="inventory-optimization"></div>', unsafe_allow_html=True)
st.subheader("📦 Inventory Optimization & Demand Planning")
st.write(
    "Use historical product demand to estimate 30-day requirements "
    "and support stock-planning decisions."
)

try:
    inventory_recommendations = pd.read_csv("inventory_recommendations.csv")
    required_inventory_cols = {
        "Product_Code",
        "Product_Name",
        "Average_Daily_Demand",
        "Estimated_30_Day_Demand",
        "Estimated_Safety_Stock",
        "Suggested_30_Day_Stock_Target"
    }

    if not required_inventory_cols.issubset(inventory_recommendations.columns):
        st.warning(
            "inventory_recommendations.csv is missing required columns. "
            "Run inventory_optimization.py again."
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Products Analyzed",
            f"{inventory_recommendations['Product_Code'].nunique():,}"
        )
        col2.metric(
            "Total Estimated 30-Day Demand",
            f"{inventory_recommendations['Estimated_30_Day_Demand'].sum():,.0f} units"
        )
        col3.metric(
            "Total Suggested Stock Target",
            f"{inventory_recommendations['Suggested_30_Day_Stock_Target'].sum():,.0f} units"
        )

        st.write("### 🏆 Top 10 Products by Estimated 30-Day Demand")
        top_inventory = inventory_recommendations.nlargest(
            10, "Estimated_30_Day_Demand"
        )
        st.bar_chart(
            top_inventory,
            x="Product_Name",
            y="Estimated_30_Day_Demand",
            horizontal=True
        )

        st.write("### 🔎 Search for a Product")
        inventory_search = st.text_input(
            "Enter a product name or product code",
            key="inventory_product_search"
        )

        if inventory_search.strip():
            mask = (
                inventory_recommendations["Product_Name"]
                .fillna("")
                .astype(str)
                .str.contains(inventory_search, case=False, na=False, regex=False)
                |
                inventory_recommendations["Product_Code"]
                .fillna("")
                .astype(str)
                .str.contains(inventory_search, case=False, na=False, regex=False)
            )
            inventory_results = inventory_recommendations[mask]
        else:
            inventory_results = inventory_recommendations.head(20)

        st.dataframe(
            inventory_results[
                [
                    "Product_Code",
                    "Product_Name",
                    "Average_Daily_Demand",
                    "Estimated_30_Day_Demand",
                    "Estimated_Safety_Stock",
                    "Suggested_30_Day_Stock_Target"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "📥 Download Inventory Recommendations",
            data=inventory_recommendations.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
            key="download_inventory_recommendations"
        )

        st.info(
            "Planning limitation: these are historical-demand estimates, not actual reorder quantities. "
            "Current stock, supplier lead times, and service-level targets are not available in the dataset. "
            "The safety-stock calculation is illustrative and should be validated before operational use."
        )
except FileNotFoundError:
    st.warning(
        "Inventory recommendations were not found. "
        "Run inventory_optimization.py first."
    )

# ---------------------------------------------------------
# LSTM 30-day forecast
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="lstm-forecast"></div>', unsafe_allow_html=True)
st.subheader("🧠 LSTM: 30-Day Sales Forecast")
# ---------------------------------------------------------
# What-If Analysis
# ---------------------------------------------------------

st.markdown("---")
st.markdown('<div id="what-if-analysis"></div>', unsafe_allow_html=True)
st.subheader("🎛️ What-If Analysis")

st.write("Explore how changes in product demand affect stock requirements.")

try:
    what_if_df = pd.read_csv("inventory_recommendations.csv")

    required_cols = [
        "Product_Code",
        "Product_Name",
        "Average_Daily_Demand",
        "Estimated_Safety_Stock"
    ]

    if all(col in what_if_df.columns for col in required_cols):

        what_if_df["Average_Daily_Demand"] = pd.to_numeric(
            what_if_df["Average_Daily_Demand"], errors="coerce"
        )

        what_if_df["Estimated_Safety_Stock"] = pd.to_numeric(
            what_if_df["Estimated_Safety_Stock"], errors="coerce"
        )

        what_if_df = what_if_df.dropna(subset=required_cols)

        if not what_if_df.empty:

            what_if_df["Product_Label"] = (
                what_if_df["Product_Name"].astype(str)
                + " ("
                + what_if_df["Product_Code"].astype(str)
                + ")"
            )

            selected_product = st.selectbox(
                "Select a product",
                what_if_df["Product_Label"].tolist(),
                key="what_if_product"
            )

            product = what_if_df[
                what_if_df["Product_Label"] == selected_product
            ].iloc[0]

            demand_change = st.slider(
                "Change in demand (%)",
                -50, 100, 0, 5
            )

            coverage_days = st.slider(
                "Stock coverage (days)",
                7, 60, 30
            )

            daily_demand = float(product["Average_Daily_Demand"])
            safety_stock = float(product["Estimated_Safety_Stock"])

            adjusted_demand = daily_demand * (1 + demand_change / 100)
            estimated_demand = adjusted_demand * coverage_days
            adjusted_safety = safety_stock * (1 + demand_change / 100)
            stock_target = estimated_demand + adjusted_safety

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Adjusted Daily Demand",
                f"{adjusted_demand:,.2f}"
            )

            col2.metric(
                "Estimated Demand",
                f"{estimated_demand:,.2f}"
            )

            col3.metric(
                "Suggested Stock Target",
                f"{stock_target:,.2f}"
            )

            result = pd.DataFrame({
                "Product": [product["Product_Name"]],
                "Demand Change (%)": [demand_change],
                "Coverage Days": [coverage_days],
                "Adjusted Daily Demand": [adjusted_demand],
                "Estimated Demand": [estimated_demand],
                "Suggested Stock Target": [stock_target]
            })

            st.dataframe(result, use_container_width=True)

            st.download_button(
                "📥 Download What-If Results",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="retailpulse_what_if_results.csv",
                mime="text/csv",
                key="download_what_if_results"
            )

        else:
            st.warning("No valid inventory products are available.")

    else:
        st.warning("Required columns are missing from the inventory file.")

except FileNotFoundError:
    st.warning("inventory_recommendations.csv was not found.")

except Exception as error:
    st.warning(f"What-If Analysis error: {error}")

try:
    lstm_df = pd.read_csv("lstm_sales_forecast.csv")
    required_lstm_cols = {"Date", "Predicted_Sales"}
    if not required_lstm_cols.issubset(lstm_df.columns):
        st.warning("The LSTM forecast CSV is missing required columns.")
    else:
        lstm_df["Date"] = pd.to_datetime(lstm_df["Date"], errors="coerce")
        lstm_df["Predicted_Sales"] = pd.to_numeric(
            lstm_df["Predicted_Sales"], errors="coerce"
        )
        lstm_df = lstm_df.dropna(subset=["Date", "Predicted_Sales"])
        st.write(
            "This forecast uses an LSTM neural network trained on historical daily sales data."
        )
        st.write("### Daily Predicted Sales")
        st.line_chart(lstm_df, x="Date", y="Predicted_Sales")

        col1, col2 = st.columns(2)
        col1.metric(
            "Average Predicted Daily Sales (LSTM)",
            f"£{lstm_df['Predicted_Sales'].mean():,.2f}"
        )
        col2.metric(
            "Total Predicted Sales (LSTM, 30 Days)",
            f"£{lstm_df['Predicted_Sales'].sum():,.2f}"
        )

        st.download_button(
            "📥 Download LSTM Forecast CSV",
            data=lstm_df.to_csv(index=False).encode("utf-8"),
            file_name="lstm_sales_forecast.csv",
            mime="text/csv",
            key="download_lstm_forecast"
        )
except FileNotFoundError:
    st.warning("LSTM forecast file not found. Run lstm_forecasting.py first.")

# ---------------------------------------------------------
# Forecast model comparison
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="model-comparison"></div>', unsafe_allow_html=True)
st.subheader("📊 Forecast Model Comparison")
st.write(
    "On the current historical test period, Prophet's reported MAE was 13,994.70 "
    "and MAPE was 23.21%; LSTM's reported MAE was 27,414.14 and MAPE was 40.94%. "
    "These are preliminary results from the existing runs, not a guarantee of future accuracy."
)
comparison_df = pd.DataFrame({
    "Model": ["Prophet", "LSTM"],
    "MAE": [13994.70, 27414.14],
    "MAPE_Percent": [23.21, 40.94]
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.caption(
    "RetailPulse demo: results depend on the supplied historical dataset. "
    "Validate models and inventory assumptions before making business decisions."
)


# ---------------------------------------------------------
# Customer churn prediction
# ---------------------------------------------------------
st.markdown("---")
st.markdown('<div id="churn-prediction"></div>', unsafe_allow_html=True)
st.subheader("⚠️ Churn Prediction & Customer Risk")
st.write(
    "This section estimates which customers may become inactive, using the saved churn model. "
    "The model was trained on historical data and its current evaluation is below the assignment target, "
    "so use these scores as a demonstration rather than a business decision."
)

try:
    import joblib

    churn_data = pd.read_csv("churn_model_dataset.csv")
    churn_model = joblib.load("churn_model.pkl")

    churn_features = [
        "Total_Orders",
        "Total_Spending",
        "Total_Units",
        "Recency",
        "Average_Order_Value",
        "Average_Units_Per_Order",
    ]

    missing_churn_columns = [
        column for column in churn_features if column not in churn_data.columns
    ]

    if missing_churn_columns:
        st.warning(
            "The churn dataset is missing required model columns: "
            + ", ".join(missing_churn_columns)
        )
    else:
        churn_view = churn_data.copy()

        # Convert model inputs to numeric values and only score valid rows.
        for column in churn_features:
            churn_view[column] = pd.to_numeric(churn_view[column], errors="coerce")

        valid_churn_rows = churn_view[churn_features].notna().all(axis=1)
        scored = churn_view.loc[valid_churn_rows].copy()

        if scored.empty:
            st.warning("No valid rows were available for churn prediction.")
        else:
            scored["Predicted_Churn_Risk"] = churn_model.predict_proba(
                scored[churn_features]
            )[:, 1]

            scored["Risk_Level"] = pd.cut(
                scored["Predicted_Churn_Risk"],
                bins=[-0.001, 0.40, 0.70, 1.001],
                labels=["Lower", "Medium", "Higher"],
            )

            high_risk_count = int((scored["Risk_Level"] == "Higher").sum())
            average_risk = float(scored["Predicted_Churn_Risk"].mean())

            risk_col1, risk_col2, risk_col3 = st.columns(3)
            risk_col1.metric("Customers Scored", f"{len(scored):,}")
            risk_col2.metric("Higher-Risk Customers", f"{high_risk_count:,}")
            risk_col3.metric("Average Predicted Risk", f"{average_risk:.1%}")

            st.write("### Highest-Risk Customers")
            display_columns = [
                column for column in [
                    "Customer_ID",
                    "Total_Orders",
                    "Total_Spending",
                    "Recency",
                    "Predicted_Churn_Risk",
                    "Risk_Level",
                ]
                if column in scored.columns
            ]
            top_risk = scored.sort_values(
                "Predicted_Churn_Risk", ascending=False
            ).head(20)

            formatted_top_risk = top_risk[display_columns].copy()
            formatted_top_risk["Predicted_Churn_Risk"] = (
                formatted_top_risk["Predicted_Churn_Risk"].map(lambda value: f"{value:.1%}")
            )
            st.dataframe(
                formatted_top_risk,
                use_container_width=True,
                hide_index=True,
            )

            risk_counts = (
                scored["Risk_Level"]
                .value_counts()
                .reindex(["Lower", "Medium", "Higher"], fill_value=0)
                .rename_axis("Risk Level")
                .reset_index(name="Number of Customers")
            )
            st.write("### Customer Risk Distribution")
            st.bar_chart(
                risk_counts,
                x="Risk Level",
                y="Number of Customers",
            )

            st.download_button(
                "📥 Download Churn Risk Predictions",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="customer_churn_risk_predictions.csv",
                mime="text/csv",
                key="download_churn_risk_predictions",
            )
except FileNotFoundError as error:
    st.warning(
        f"Churn file not found: {error}. Make sure churn_model.pkl and "
        "churn_model_dataset.csv are in the RetailPulse folder."
    )
except Exception as error:
    st.warning(
        "The churn section could not load the saved model or score customers. "
        f"Details: {error}"
    )
