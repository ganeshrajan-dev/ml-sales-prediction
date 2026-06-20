"""
DATA VALIDATION
===============
What: Checks data quality before training
Why:  Bad data = bad model. Catch problems early.
Input:  Cleaned CSV from data_ingestion.py
Output: Validated CSV + validation report
Gate:   If quality < 80%, pipeline STOPS here
"""

import pandas as pd
import yaml
import json
import os
import sys
from datetime import datetime


def load_config():
    """Load configuration from YAML file"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_validation():
    """Main validation function"""

    print("=" * 60)
    print("STAGE 2: DATA VALIDATION")
    print("=" * 60)

    # Load config
    config = load_config()
    input_path = config["data"]["raw_local"]
    output_path = config["data"]["validated_local"]

    # -------------------------------------------
    # STEP 1: Load cleaned data from previous stage
    # -------------------------------------------
    print(f"\n[Step 1] Loading data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['Invoice Date'])
    print(f"         Rows: {len(df)}")

    # We'll track all check results
    checks = []
    passed = 0
    failed = 0

    # -------------------------------------------
    # CHECK 1: Schema — Are all required columns present?
    #
    # Why: If data source changes and a column is missing,
    #      feature engineering will crash with a confusing error.
    #      Better to catch it here with a clear message.
    # -------------------------------------------
    print("\n[Step 2] Running validation checks...")

    expected_columns = [
        'Retailer', 'Retailer ID', 'Invoice Date', 'Region',
        'State', 'City', 'Product', 'Price per Unit', 'Units Sold',
        'Total Sales', 'Operating Profit', 'Sales Method'
    ]
    missing = set(expected_columns) - set(df.columns)
    check_pass = len(missing) == 0
    checks.append({"check": "schema", "passed": bool(check_pass),
                   "detail": f"Missing: {list(missing)}" if not check_pass else "All columns present"})
    print(f"         {'PASS' if check_pass else 'FAIL'} — Schema check: {len(missing)} missing columns")
    passed += check_pass
    failed += (not check_pass)

    # -------------------------------------------
    # CHECK 2: Nulls — Are there too many empty values?
    #
    # Why: A few nulls (< 5%) is okay — we drop them.
    #      But if 50% of data is null, something is seriously wrong.
    # -------------------------------------------
    null_pct = df.isnull().sum() / len(df) * 100
    max_null = null_pct.max()
    check_pass = max_null < 5.0
    checks.append({"check": "null_threshold", "passed": bool(check_pass),
                   "detail": f"Max null: {max_null:.2f}% (limit: 5%)"})
    print(f"         {'PASS' if check_pass else 'FAIL'} — Null check: max {max_null:.2f}% (limit 5%)")
    passed += check_pass
    failed += (not check_pass)

    # -------------------------------------------
    # CHECK 3: No negative values in numeric columns
    #
    # Why: Price can't be -$50. Units sold can't be -100.
    #      Negative values = data corruption or processing error.
    # -------------------------------------------
    numeric_cols = ['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit']
    negatives = {}
    for col in numeric_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            negatives[col] = int(neg_count)
    check_pass = len(negatives) == 0
    checks.append({"check": "no_negatives", "passed": bool(check_pass),
                   "detail": f"Negatives: {negatives}" if negatives else "No negatives"})
    print(f"         {'PASS' if check_pass else 'FAIL'} — Negative values: {negatives if negatives else 'none'}")
    passed += check_pass
    failed += (not check_pass)

    # -------------------------------------------
    # CHECK 4: Target column has only expected values
    #
    # Why: Our model expects 3 classes — Online, In-store, Outlet.
    #      If a new value appears (e.g., "Wholesale"), model will crash.
    # -------------------------------------------
    expected_targets = {'In-store', 'Online', 'Outlet'}
    actual_targets = set(df['Sales Method'].unique())
    check_pass = actual_targets.issubset(expected_targets)
    checks.append({"check": "target_values", "passed": bool(check_pass),
                   "detail": f"Found: {list(actual_targets)}"})
    print(f"         {'PASS' if check_pass else 'FAIL'} — Target values: {actual_targets}")
    passed += check_pass
    failed += (not check_pass)

    # -------------------------------------------
    # CHECK 5: Duplicate rows
    #
    # Why: If same transaction appears 100 times, model learns
    #      that specific case too well (overfitting).
    #      Some duplicates are okay (< 10%), too many is bad.
    # -------------------------------------------
    dup_count = df.duplicated().sum()
    dup_pct = dup_count / len(df) * 100
    check_pass = dup_pct < 10.0
    checks.append({"check": "duplicates", "passed": bool(check_pass),
                   "detail": f"Duplicates: {dup_count} ({dup_pct:.1f}%)"})
    print(f"         {'PASS' if check_pass else 'FAIL'} — Duplicates: {dup_count} ({dup_pct:.1f}%)")
    passed += check_pass
    failed += (not check_pass)

    # -------------------------------------------
    # CHECK 6: Minimum data volume
    #
    # Why: Model needs enough data to learn patterns.
    #      With only 100 rows, XGBoost can't learn reliably.
    #      We need at least 1000 rows.
    # -------------------------------------------
    min_rows = 1000
    check_pass = len(df) >= min_rows
    checks.append({"check": "min_volume", "passed": bool(check_pass),
                   "detail": f"Rows: {len(df)} (minimum: {min_rows})"})
    print(f"         {'PASS' if check_pass else 'FAIL'} — Data volume: {len(df)} rows (min: {min_rows})")
    passed += check_pass
    failed += (not check_pass)

    # -------------------------------------------
    # QUALITY SCORE
    # -------------------------------------------
    total_checks = passed + failed
    quality_score = passed / total_checks
    print(f"\n[Step 3] Quality Score: {quality_score:.0%} ({passed}/{total_checks} checks passed)")

    # -------------------------------------------
    # STEP 4: Clean — drop nulls and duplicates
    # -------------------------------------------
    print(f"\n[Step 4] Cleaning...")
    before = len(df)
    df = df.dropna()
    print(f"         Dropped nulls: {before} → {len(df)} rows")

    before = len(df)
    df = df.drop_duplicates()
    print(f"         Dropped duplicates: {before} → {len(df)} rows")

    # -------------------------------------------
    # STEP 5: Save validated data
    # -------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[Step 5] Saved validated data: {output_path}")
    print(f"         Final: {len(df)} rows")

    # -------------------------------------------
    # STEP 6: Save validation report
    # -------------------------------------------
    report = {
        "stage": "data_validation",
        "timestamp": datetime.now().isoformat(),
        "input": input_path,
        "output": output_path,
        "records_in": before,
        "records_out": len(df),
        "quality_score": round(quality_score, 4),
        "checks": checks,
        "status": "PASSED" if quality_score >= 0.8 else "FAILED"
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"         Report saved: artifacts/validation_report.json")

    if quality_score < 0.8:
        print(f"\n*** PIPELINE FAILED: Quality {quality_score:.0%} < 80% ***")
        print(f"*** Fix the data issues and re-run ***")
        sys.exit(1)  # Exit code 1 = failure (Jenkins will see this)
    
    print(f"\n{'=' * 60}")
    print(f"DATA VALIDATION PASSED — Quality: {quality_score:.0%}")
    print(f"{'=' * 60}")

    return df


if __name__ == "__main__":
    run_validation()
