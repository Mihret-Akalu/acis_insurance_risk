import pandas as pd
import sys

infile = sys.argv[1]  # input CSV
outfile = sys.argv[2] # output processed file

# Load dataset
df = pd.read_csv(infile, parse_dates=['TransactionMonth'])

# Example processing
df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']

# Save processed data as parquet
df.to_parquet(outfile, index=False)
