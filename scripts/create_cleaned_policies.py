import pandas as pd
import os

# Load the processed parquet file
input_file = "data/processed/insurance.parquet"

if not os.path.exists(input_file):
    raise FileNotFoundError("ERROR: insurance.parquet does not exist. Run data_prepare.py first.")

df = pd.read_parquet(input_file)

# -----------------------------
# Create fields needed for Task 3
# -----------------------------

# 1. Claim flag (1 if charges > 0)
df["claim_flag"] = (df["charges"] > 0).astype(int)

# 2. Severity – same as charges
df["severity"] = df["charges"]

# 3. Margin – placeholder since no premium/claims
df["margin"] = df["charges"] * -0.25   # example proxy

# 4. Segment – split by BMI into A/B groups
df["Segment"] = pd.qcut(df["bmi"], q=2, labels=["A", "B"])

# -----------------------------
# Save cleaned dataset
# -----------------------------
output_file = "data/processed/cleaned_policies.csv"
df.to_csv(output_file, index=False)

print(f"SUCCESS: created {output_file} with {len(df)} rows")
