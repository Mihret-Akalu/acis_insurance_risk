import pandas as pd
import sys
import os

# Input and output files from command line
if len(sys.argv) != 3:
    print("Usage: python data_prepare.py <input_csv> <output_parquet>")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

# Check if input file exists
if not os.path.exists(infile):
    print(f"Error: input file {infile} does not exist!")
    sys.exit(1)

# Load the CSV
try:
    # Try to parse 'TransactionMonth' if it exists
    df = pd.read_csv(infile)
    if 'TransactionMonth' in df.columns:
        df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# Compute LossRatio if TotalPremium and TotalClaims exist
if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
else:
    print("Warning: TotalPremium or TotalClaims column not found. Skipping LossRatio computation.")

# Save processed data as parquet
try:
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    df.to_parquet(outfile, index=False)
    print(f"Processed data saved to {outfile}")
except Exception as e:
    print(f"Error saving parquet: {e}")
    sys.exit(1)
