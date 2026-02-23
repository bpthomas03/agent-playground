import sys
import os
import pandas as pd

# Input and Output paths
input_csv = "/app/data/clean_dreams.csv"
output_pkl = "/app/outputs/run_2026-02-23T15-44-25Z/step_1_corpus_df.pkl"

def main():
    # Check if input file exists
    if not os.path.isfile(input_csv):
        print(f"Input file not found: {input_csv}", file=sys.stderr)
        sys.exit(1)
    
    # Attempt to load the CSV into a DataFrame
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading CSV '{input_csv}': {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Basic validation: DataFrame should not be empty
    if df.empty:
        print(f"Loaded DataFrame is empty from '{input_csv}'", file=sys.stderr)
        sys.exit(1)
    
    # Attempt to save as pickle
    try:
        os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
        df.to_pickle(output_pkl)
    except Exception as e:
        print(f"Error saving pickle to '{output_pkl}': {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Wrote DataFrame with {len(df)} rows to {output_pkl}")

if __name__ == "__main__":
    main()