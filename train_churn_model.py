import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# Step 1: Load dataset
df = pd.read_csv("churn_model_dataset.csv")

# Step 2: Select features
features = [
    "Total_Orders",
    "Total_Spending",
    "Total_Units",
    "Recency",
    "Average_Order_Value",
    "Average_Units_Per_Order"
]

X = df[features]
y = df["Churn"]

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

# Step 4: Create Logistic Regression pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Step 7: Evaluate
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))

print(
    "\nROC-AUC:",
    round(roc_auc_score(y_test, probabilities), 4)
)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Step 8: Calculate top-20% precision
results = pd.DataFrame({
    "Actual_Churn": y_test.to_numpy(),
    "Churn_Probability": probabilities
})

top_20_count = max(1, int(len(results) * 0.20))

top_customers = results.nlargest(
    top_20_count,
    "Churn_Probability"
)

precision_top_20 = top_customers["Actual_Churn"].mean()

print(
    "\nPrecision among top 20% highest-risk customers:",
    round(precision_top_20, 4)
)

# Step 9: Save model
joblib.dump(model, "churn_model.pkl")

print("\nLogistic Regression model saved as churn_model.pkl")