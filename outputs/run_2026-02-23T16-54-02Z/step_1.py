import sys
import os
import pandas as pd

INPUT_PATH = '/app/data/clean_dreams.csv'
OUTPUT_PATH = '/app/outputs/run_2026-02-23T16-54-02Z/step_1_corpus_df.pkl'

def main():
    try:
        if not os.path.isfile(INPUT_PATH):
            print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
            sys.exit(1)
        # Read the CSV
        try:
            df = pd.read_csv(INPUT_PATH)
        except Exception as e:
            print(f"Error reading CSV: {e}", file=sys.stderr)
            sys.exit(1)

        # Basic validation: ensure it's not empty
        if df.empty:
            print("Loaded DataFrame is empty.", file=sys.stderr)
            sys.exit(1)
        
        # Save DataFrame as pickle
        try:
            df.to_pickle(OUTPUT_PATH)
        except Exception as e:
            print(f"Error saving DataFrame to pickle: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"Wrote DataFrame with {len(df)} rows to {OUTPUT_PATH}")

    except Exception as e:
        print(f"Unhandled exception: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()