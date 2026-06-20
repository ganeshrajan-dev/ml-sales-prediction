"""
MODEL REGISTRY (SageMaker)
==========================
What: Registers approved model in SageMaker Model Registry
Why:  Production needs a single source of truth for "which model is LIVE"
Input:  Model artifacts + evaluation report (must be PASSED)
Output: Model registered in SageMaker with version and status

SageMaker Registry gives us:
    - Version control for models
    - Approval workflow (PendingApproval → Approved)
    - IAM security (who can deploy)
    - Audit trail (CloudTrail logs)
"""

import boto3
import json
import yaml
import os
import time
from datetime import datetime


def load_config():
    """Load configuration from YAML file"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_registration():
    """Register model in SageMaker Model Registry"""

    print("=" * 60)
    print("STAGE 6: MODEL REGISTRY (SageMaker)")
    print("=" * 60)

    config = load_config()
    model_version = config["model"]["version"]
    bucket = config["aws"]["s3_bucket"]
    region = config["aws"]["region"]

    # -------------------------------------------
    # STEP 1: Check evaluation passed
    #
    # Only register if model passed quality gate.
    # Never put a failed model in registry.
    # -------------------------------------------
    print("\n[Step 1] Checking evaluation status...")

    with open("artifacts/evaluation_report.json") as f:
        eval_report = json.load(f)

    if eval_report["status"] != "PASSED":
        print("         EVALUATION FAILED — Cannot register model")
        print("         Fix the model and re-run training + evaluation")
        return

    accuracy = eval_report["metrics"]["accuracy"]
    f1 = eval_report["metrics"]["f1_weighted"]
    print(f"         Evaluation: PASSED (accuracy={accuracy}, f1={f1})")

    # -------------------------------------------
    # STEP 2: Upload model artifacts to S3
    #
    # SageMaker Registry needs model files in S3.
    # We upload: model.pkl, encoders.pkl, feature_config.json
    # as a tar.gz package (SageMaker standard format)
    # -------------------------------------------
    print("\n[Step 2] Packaging and uploading model to S3...")

    import tarfile

    # Create model.tar.gz (SageMaker expects this format)
    tar_path = "model/model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add("model/model.pkl", arcname="model.pkl")
        tar.add("model/encoders.pkl", arcname="encoders.pkl")
        tar.add("model/feature_config.json", arcname="feature_config.json")
        tar.add("model/drift_baseline.json", arcname="drift_baseline.json")

    print(f"         Packaged: {tar_path}")

    # Upload to S3
    s3_client = boto3.client("s3", region_name=region)
    s3_model_key = f"model-registry/{model_version}/model.tar.gz"

    s3_client.upload_file(tar_path, bucket, s3_model_key)
    model_s3_uri = f"s3://{bucket}/{s3_model_key}"
    print(f"         Uploaded to: {model_s3_uri}")

    # -------------------------------------------
    # STEP 3: Create Model Package Group (if not exists)
    #
    # Model Package Group = container for all versions
    # Like creating a "repository" for this model
    # Only needs to be done ONCE (first time)
    # -------------------------------------------
    print("\n[Step 3] Creating/checking Model Package Group...")

    sm_client = boto3.client("sagemaker", region_name=region)
    group_name = "adidas-sales-predictor"

    try:
        sm_client.describe_model_package_group(
            ModelPackageGroupName=group_name
        )
        print(f"         Group exists: {group_name}")
    except sm_client.exceptions.ClientError:
        # Group doesn't exist — create it
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Adidas sales channel prediction model (XGBoost)"
        )
        print(f"         Created group: {group_name}")

    # -------------------------------------------
    # STEP 4: Register Model Package (new version)
    #
    # This creates a new VERSION in the registry.
    # We include:
    #   - Where the model files are (S3)
    #   - What container to use for inference
    #   - Model metrics (for tracking)
    #   - Approval status
    # -------------------------------------------
    print("\n[Step 4] Registering model version...")

    # Get the XGBoost inference container image URI
    # (This is the AWS-provided container that knows how to serve XGBoost models)
    from sagemaker import image_uris
    inference_image = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.7-1"
    )

    # Create model package
    response = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelPackageDescription=f"XGBoost v{model_version} - Accuracy: {accuracy}, F1: {f1}",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": inference_image,
                    "ModelDataUrl": model_s3_uri
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"]
        },
        ModelApprovalStatus="Approved",  # Auto-approve since evaluation passed
        CustomerMetadataProperties={
            "accuracy": str(accuracy),
            "f1_score": str(f1),
            "model_version": model_version,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "algorithm": "XGBClassifier"
        }
    )

    model_package_arn = response["ModelPackageArn"]
    print(f"         Registered: {model_package_arn}")
    print(f"         Status: Approved")

    # -------------------------------------------
    # STEP 5: Save registration report
    # -------------------------------------------
    print("\n[Step 5] Saving registration report...")

    report = {
        "stage": "model_registration",
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "model_package_group": group_name,
        "model_package_arn": model_package_arn,
        "model_s3_uri": model_s3_uri,
        "approval_status": "Approved",
        "metrics": eval_report["metrics"],
        "status": "SUCCESS"
    }

    with open("artifacts/registration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"         Report: artifacts/registration_report.json")

    # -------------------------------------------
    # STEP 6: Verify registration
    # -------------------------------------------
    print("\n[Step 6] Verifying registration...")

    # List all versions in the group
    versions = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        SortBy="CreationTime",
        SortOrder="Descending"
    )

    print(f"\n         Registry: {group_name}")
    print(f"         {'Version':<10} {'Status':<15} {'Created'}")
    print(f"         {'-'*50}")
    for pkg in versions["ModelPackageSummaryList"]:
        ver = pkg["ModelPackageArn"].split("/")[-1]
        status = pkg["ModelApprovalStatus"]
        created = pkg["CreationTime"].strftime("%Y-%m-%d %H:%M")
        print(f"         {ver:<10} {status:<15} {created}")

    print(f"\n{'=' * 60}")
    print(f"MODEL REGISTERED SUCCESSFULLY")
    print(f"  Group: {group_name}")
    print(f"  Version ARN: {model_package_arn}")
    print(f"  Status: Approved")
    print(f"  Artifacts: {model_s3_uri}")
    print(f"{'=' * 60}")

    return model_package_arn


if __name__ == "__main__":
    run_registration()