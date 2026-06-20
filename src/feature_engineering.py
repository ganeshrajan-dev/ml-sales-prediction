"""
FEATURE ENGINEERING
===================
What: Converts raw data into numerical features model can learn from
Why:  XGBoost only understands numbers. Text and dates must be transformed.
Input:  Validated CSV from data_validation.py
Output: Processed CSV with features + encoders saved for serving
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import yaml
import json
import os
from datetime import datetime


def load_config():
    """Load configuration from YAML file"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_feature_engineering():
    """Main feature engineering function"""

    print("=" * 60)
    print("STAGE 3: FEATURE ENGINEERING")
    print("=" * 60)

    # Load config
    config = load_config()
    input_path = config["data"]["validated_local"]
    output_path = config["data"]["processed_local"]

    # -------------------------------------------
    # STEP 1: Load validated data
    # -------------------------------------------
    print(f"\n[Step 1] Loading validated data: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['Invoice Date'])
    print(f"         Rows: {len(df)}, Columns: {len(df.columns)}")

    # -------------------------------------------
    # STEP 2: Create TIME features from Invoice Date
    # -------------------------------------------
    print("\n[Step 2] Creating time features from 'Invoice Date'...")

    df['Month'] = df['Invoice Date'].dt.month
    df['DayOfWeek'] = df['Invoice Date'].dt.dayofweek
    df['Quarter'] = df['Invoice Date'].dt.quarter
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    print("         Created: Month, DayOfWeek, Quarter, IsWeekend")
    print(f"         Example: Date 2021-06-17 → Month=6, DayOfWeek=3(Thu), Quarter=2, IsWeekend=0")

    # -------------------------------------------
    # STEP 3: Create BUSINESS features
    # -------------------------------------------
    print("\n[Step 3] Creating business features...")

    df['Profit_Margin'] = (df['Operating Profit'] / df['Total Sales'] * 100).round(2)
    df['Revenue_Per_Unit'] = (df['Total Sales'] / df['Units Sold']).round(2)

    print("         Created: Profit_Margin, Revenue_Per_Unit")
    print(f"         Example: Profit $1257 / Sales $2245 = Margin 56.0%")

    # -------------------------------------------
    # STEP 4: Encode CATEGORICAL features
    # -------------------------------------------
    print("\n[Step 4] Encoding categorical features...")

    categorical_cols = config["features"]["categorical_columns"]
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"         {col}: {len(le.classes_)} classes → {dict(zip(le.classes_, range(len(le.classes_))))}")

    # -------------------------------------------
    # STEP 5: Encode TARGET variable (Sales Method)
    # -------------------------------------------
    print("\n[Step 5] Encoding target variable (Sales Method)...")

    target_encoder = LabelEncoder()
    df['Target'] = target_encoder.fit_transform(df['Sales Method'])
    label_encoders['Target'] = target_encoder

    target_mapping = dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))
    print(f"         Target mapping: {target_mapping}")

    # -------------------------------------------
    # STEP 6: Select final feature columns
    # -------------------------------------------
    print("\n[Step 6] Selecting final features...")

    feature_columns = [
        'Retailer_enc', 'Region_enc', 'State_enc', 'City_enc', 'Product_enc',
        'Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit',
        'Profit_Margin', 'Revenue_Per_Unit',
        'Month', 'DayOfWeek', 'Quarter', 'IsWeekend'
    ]

    print(f"         Total features: {len(feature_columns)}")
    for i, col in enumerate(feature_columns, 1):
        print(f"           {i:2d}. {col}")

    # -------------------------------------------
    # STEP 7: Save outputs
    # -------------------------------------------
    print("\n[Step 7] Saving outputs...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"         Processed data: {output_path}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(label_encoders, "model/encoders.pkl")
    print(f"         Encoders: model/encoders.pkl")

    feature_config = {
        "feature_columns": feature_columns,
        "target_column": "Target",
        "target_mapping": target_mapping,
        "categorical_columns": categorical_cols,
        "categorical_encoded": [c + '_enc' for c in categorical_cols],
        "numerical_columns": [
            'Price per Unit', 'Units Sold', 'Total Sales',
            'Operating Profit', 'Profit_Margin', 'Revenue_Per_Unit'
        ],
        "time_features": ['Month', 'DayOfWeek', 'Quarter', 'IsWeekend']
    }

    with open("model/feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"         Feature config: model/feature_config.json")

    # -------------------------------------------
    # STEP 8: Save drift baseline
    # -------------------------------------------
    print("\n[Step 8] Saving drift baseline...")

    drift_baseline = {}

    for col in feature_columns:
        drift_baseline[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max())
        }

    with open("model/drift_baseline.json", "w") as f:
        json.dump(drift_baseline, f, indent=2)

    print(f"         Drift baseline: model/drift_baseline.json")

    # -------------------------------------------
    # STEP 9: Report
    # -------------------------------------------
    report = {
        "stage": "feature_engineering",
        "timestamp": datetime.now().isoformat(),
        "features_count": len(feature_columns),
        "features": feature_columns,
        "target_mapping": target_mapping,
        "records": len(df),
        "status": "SUCCESS"
    }

    with open("artifacts/feature_engineering_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"FEATURE ENGINEERING COMPLETE — {len(feature_columns)} features created")
    print(f"{'=' * 60}")

    return df, feature_columns


if __name__ == "__main__":
    run_feature_engineering()