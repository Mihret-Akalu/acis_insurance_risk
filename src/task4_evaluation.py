# -------------------------
# Task 4: Claim Severity Modeling
# -------------------------

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_policies.csv")
print("Dataset loaded. Sample:")
print(df.head())

# 2. Prepare features and target
# Drop non-numeric or non-feature columns
X = df.drop(columns=['claim_flag', 'severity', 'margin', 'Segment'])
y = df['severity']

# Optional: one-hot encode categorical columns
cat_cols = X.select_dtypes(include='object').columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest Regressor
reg = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
reg.fit(X_train, y_train)

# Save model
os.makedirs("models/saved_models", exist_ok=True)
joblib.dump(reg, "models/saved_models/claim_severity_model.pkl")

# 5. Evaluate model
y_pred = reg.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Severity RMSE: {rmse:.2f}")
print(f"Severity R2: {r2:.4f}")

# 6. SHAP interpretability
explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_test)

# Summary plot
os.makedirs("reports/figures", exist_ok=True)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("reports/figures/shap_regression.png", dpi=300)
plt.close()

# 7. Save evaluation summary
summary = f"""
===============================
     Task 4 Model Evaluation
===============================

Regression (Claim Severity):
- RMSE: {rmse:.2f}
- R2: {r2:.4f}

===============================
"""

os.makedirs("models/evaluation_reports", exist_ok=True)
with open("models/evaluation_reports/task4_model_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print("Task 4 evaluation summary saved.")
print("SHAP plot saved at reports/figures/shap_regression.png")
