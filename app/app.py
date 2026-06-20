
"""
MODEL SERVING API
=================
What: Flask REST API that serves predictions from trained XGBoost model
Why:  Model needs an HTTP interface for other services to call
Endpoints:
    /predict    - Main prediction endpoint
    /health     - Kubernetes liveness/readiness probe
    /metrics    - Prometheus monitoring metrics
    /model/info - Model metadata
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import time
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ============================================================
# APP INITIALIZATION
# ============================================================

app = Flask(__name__)

# -------------------------------------------
# Load model artifacts at startup
#
# These are loaded ONCE when the container starts.
# Not on every request — that would be too slow.
# They stay in RAM for fast predictions.
# -------------------------------------------

MODEL_DIR = os.environ.get("MODEL_DIR", "model")

print(f"Loading model from: {MODEL_DIR}")

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))

with open(os.path.join(MODEL_DIR, "feature_config.json")) as f:
    feature_config = json.load(f)

with open(os.path.join(MODEL_DIR, "drift_baseline.json")) as f:
    drift_baseline = json.load(f)

# Extract target encoder from encoders dict
target_encoder = encoders["Target"]
target_classes = list(feature_config["target_mapping"].keys())

MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")

print(f"Model loaded successfully! Version: {MODEL_VERSION}")

# ============================================================
# PROMETHEUS METRICS
#
# These counters/histograms are updated on every prediction.
# Prometheus scrapes /metrics endpoint every 15 seconds.
# Grafana visualizes them as dashboards.
# ============================================================

# Count total predictions (labeled by predicted class)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total number of predictions made",
    ["predicted_class", "model_version"]
)

# Track prediction latency (how long each prediction takes)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time taken to make a prediction",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Track data drift score
DATA_DRIFT_SCORE = Gauge(
    "model_data_drift_score",
    "Current data drift score (0=no drift, 1=severe drift)"
)

# Track errors
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Total prediction errors",
    ["error_type"]
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Kubernetes calls this every 10 seconds:
        - If returns 200 → pod is healthy
        - If returns 500 or timeout → pod is dead, K8s restarts it

    Also used for readiness probe:
        - Returns 200 only if model is loaded and ready to serve
    """
    return jsonify({
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.

    Expects JSON body:
    {
        "retailer": "Foot Locker",
        "region": "Midwest",
        "state": "Ohio",
        "city": "Columbus",
        "product": "Men's Athletic Footwear",
        "price_per_unit": 75.0,
        "units_sold": 300,
        "total_sales": 22500.0,
        "operating_profit": 9000.0,
        "month": 7,
        "day_of_week": 3,
        "quarter": 3,
        "is_weekend": 0
    }

    Returns:
    {
        "predicted_sales_method": "Outlet",
        "confidence": {"In-store": 0.20, "Online": 0.08, "Outlet": 0.72},
        "model_version": "v1",
        "drift_score": 0.12,
        "latency_ms": 5.2
    }
    """
    start_time = time.time()

    try:
        data = request.get_json()

        # -------------------------------------------
        # Encode categorical features
        # Same encoders used during training
        # "Foot Locker" → 1, "Midwest" → 0, etc.
        # -------------------------------------------
        retailer_enc = encoders['Retailer'].transform([data['retailer']])[0]
        region_enc = encoders['Region'].transform([data['region']])[0]
        state_enc = encoders['State'].transform([data['state']])[0]
        city_enc = encoders['City'].transform([data['city']])[0]
        product_enc = encoders['Product'].transform([data['product']])[0]

        # -------------------------------------------
        # Build feature array
        # MUST be in SAME ORDER as training features
        # -------------------------------------------
        features = np.array([[
            retailer_enc,           # Retailer_enc
            region_enc,             # Region_enc
            state_enc,              # State_enc
            city_enc,               # City_enc
            product_enc,            # Product_enc
            data['price_per_unit'], # Price per Unit
            data['units_sold'],     # Units Sold
            data['total_sales'],    # Total Sales
            data['operating_profit'],  # Operating Profit
            data.get('profit_margin', data['operating_profit'] / data['total_sales'] * 100),  # Profit_Margin
            data.get('revenue_per_unit', data['total_sales'] / data['units_sold']),  # Revenue_Per_Unit
            data['month'],          # Month
            data['day_of_week'],    # DayOfWeek
            data['quarter'],        # Quarter
            data['is_weekend']      # IsWeekend
        ]])

        # -------------------------------------------
        # Make prediction
        # -------------------------------------------
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        predicted_class = target_encoder.inverse_transform([prediction])[0]
        confidence = {
            cls: round(float(prob), 4)
            for cls, prob in zip(target_classes, probabilities)
        }

        # -------------------------------------------
        # Calculate drift score
        # Compare input features against training baseline
        # -------------------------------------------
        drift_score = _calculate_drift(features[0])
        DATA_DRIFT_SCORE.set(drift_score)

        # -------------------------------------------
        # Update Prometheus metrics
        # -------------------------------------------
        latency = time.time() - start_time
        PREDICTION_COUNT.labels(
            predicted_class=predicted_class,
            model_version=MODEL_VERSION
        ).inc()
        PREDICTION_LATENCY.observe(latency)

        return jsonify({
            "predicted_sales_method": predicted_class,
            "confidence": confidence,
            "model_version": MODEL_VERSION,
            "drift_score": round(drift_score, 4),
            "drift_warning": drift_score > 0.5,
            "latency_ms": round(latency * 1000, 2)
        })

    except KeyError as e:
        PREDICTION_ERRORS.labels(error_type="missing_field").inc()
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400

    except ValueError as e:
        PREDICTION_ERRORS.labels(error_type="invalid_value").inc()
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="internal_error").inc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    Prometheus metrics endpoint.

    Prometheus server scrapes this every 15 seconds.
    Returns all metrics in Prometheus text format.

    Example output:
        model_predictions_total{predicted_class="Online",model_version="v1"} 1523
        model_prediction_latency_seconds_bucket{le="0.01"} 1400
        model_data_drift_score 0.15
    """
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/model/info", methods=["GET"])
def model_info():
    """
    Returns model metadata.
    Useful for: dashboards, debugging, version checking
    """
    return jsonify({
        "model_version": MODEL_VERSION,
        "algorithm": "XGBClassifier",
        "features": feature_config["feature_columns"],
        "n_features": len(feature_config["feature_columns"]),
        "target_classes": target_classes,
        "target_mapping": feature_config["target_mapping"]
    })


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _calculate_drift(features):
    """
    Simple drift detection.

    Checks how many features are outside 2 standard deviations
    of the training data distribution.

    Score:
        0.0 = no drift (all features within normal range)
        1.0 = severe drift (all features are abnormal)

    Production would use: KS-test, PSI (Population Stability Index),
    or Evidently AI for more sophisticated drift detection.
    """
    feature_names = feature_config["feature_columns"]
    drift_count = 0

    for i, fname in enumerate(feature_names):
        if fname in drift_baseline:
            stats = drift_baseline[fname]
            value = features[i]
            lower = stats["mean"] - 2 * stats["std"]
            upper = stats["mean"] + 2 * stats["std"]
            if value < lower or value > upper:
                drift_count += 1

    return drift_count / len(feature_names)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    # Development only — production uses gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)