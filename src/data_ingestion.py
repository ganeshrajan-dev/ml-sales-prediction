"""
DATA INGESTION
==============
What: Loads raw data from S3, does basic cleaning
Why:  Raw data has $, commas, typos — model can't read dirty data
Input:  Raw CSV from S3
Output: Cleaned CSV saved to data/raw/
"""

import os
import json
from datetime import datetime
import pandas as pd
import yaml


def load_config():
    """Load configuration from YAML file"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_ingestion():
    """Main ingestion function"""

    print("=" * 60)
    print("STAGE 1: DATA INGESTION")
    print("=" * 60)

    # Load config
    config = load_config()
    source_path = config["data"]["raw_source"]
    output_path = config["data"]["raw_local"]

    # -------------------------------------------
    # STEP 1: Load data from S3
    # -------------------------------------------
    print(f"\n[Step 1] Loading data from: {source_path}")
    df = pd.read_csv(source_path)
    print(f"         Rows loaded: {len(df)}")
    print(f"         Columns: {df.columns.tolist()}")

    # -------------------------------------------
    # STEP 2: Clean 'Price per Unit' column
    # Raw data has: "$103.00 " (dollar sign + space)
    # We need:     103.00 (just the number)
    # -------------------------------------------
    print("\n[Step 2] Cleaning 'Price per Unit'...")
    df['Price per Unit'] = df['Price per Unit'].str.replace('$', '', regex=False)
    df['Price per Unit'] = df['Price per Unit'].str.strip()
    df['Price per Unit'] = pd.to_numeric(df['Price per Unit'], errors='coerce')
    print("         Removed $ and spaces, converted to number")

    # -------------------------------------------
    # STEP 3: Clean 'Total Sales' column
    # Raw data has: "2,245" (comma separated)
    # We need:     2245 (just the number)
    # -------------------------------------------
    print("\n[Step 3] Cleaning 'Total Sales'...")
    df['Total Sales'] = df['Total Sales'].str.replace(',', '', regex=False)
    df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')
    print("         Removed commas, converted to number")

    # -------------------------------------------
    # STEP 4: Clean 'Units Sold' column
    # -------------------------------------------
    print("\n[Step 4] Cleaning 'Units Sold'...")
    df['Units Sold'] = df['Units Sold'].str.replace(',', '', regex=False)
    df['Units Sold'] = pd.to_numeric(df['Units Sold'], errors='coerce')
    print("         Converted to number")

    # -------------------------------------------
    # STEP 5: Clean 'Operating Profit' column
    # Raw data has: "$1,257" (dollar + comma)
    # We need:     1257
    # -------------------------------------------
    print("\n[Step 5] Cleaning 'Operating Profit'...")
    df['Operating Profit'] = df['Operating Profit'].str.replace('$', '', regex=False)
    df['Operating Profit'] = df['Operating Profit'].str.replace(',', '', regex=False)
    df['Operating Profit'] = pd.to_numeric(df['Operating Profit'], errors='coerce')
    print("         Removed $ and commas, converted to number")

    # -------------------------------------------
    # STEP 6: Fix typo in Product column
    # "Men's aparel" → "Men's Apparel"
    # -------------------------------------------
    print("\n[Step 6] Fixing product name typo...")
    df['Product'] = df['Product'].replace("Men's aparel", "Men's Apparel")
    print("         Fixed: 'Men's aparel' → 'Men's Apparel'")

    # -------------------------------------------
    # STEP 7: Parse Invoice Date to datetime
    # -------------------------------------------
    print("\n[Step 7] Parsing dates...")
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%m/%d/%Y')
    print("         Converted to datetime format")

    # -------------------------------------------
    # STEP 8: Save cleaned data
    # -------------------------------------------
    print(f"\n[Step 8] Saving cleaned data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"         Saved to: {output_path}")
    print(f"         Final shape: {df.shape}")

    # -------------------------------------------
    # STEP 9: Generate ingestion report
    # (For audit trail — who ran what, when, what happened)
    # -------------------------------------------
    report = {
        "stage": "data_ingestion",
        "timestamp": datetime.now().isoformat(),
        "source": source_path,
        "output": output_path,
        "records": len(df),
        "columns": df.columns.tolist(),
        "nulls": df.isnull().sum().astype(int).to_dict(),  # Fixed: Cast values to native Python integers
        "status": "SUCCESS"
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/ingestion_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n         Report saved: artifacts/ingestion_report.json")

    print(f"\n{'=' * 60}")
    print(f"DATA INGESTION COMPLETE — {len(df)} records cleaned")
    print(f"{'=' * 60}")

    return df


# This runs when you execute: python src/data_ingestion.py
if __name__ == "__main__":
    run_ingestion()
