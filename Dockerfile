# ============================================================
# STAGE 1: Builder
# Purpose: Install all Python dependencies
# Why separate stage? So final image is smaller (no build tools)
# ============================================================
FROM python:3.10-slim AS builder

# Set working directory inside container
WORKDIR /build

# Copy only requirements first (Docker caches this layer)
# If requirements don't change, Docker reuses cached dependencies
# This makes rebuilds MUCH faster
COPY app/requirements.txt .

# Install dependencies into a separate folder (/install)
# --no-cache-dir: don't store pip cache (saves space)
# --prefix=/install: install to custom location (we'll copy only this to final image)
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ============================================================
# STAGE 2: Production
# Purpose: Minimal final image with only what's needed to RUN
# No build tools, no cache, no unnecessary files
# ============================================================
FROM python:3.10-slim

# Security: Create non-root user
# Why? If someone exploits your app, they get limited permissions
# Never run production containers as root
RUN useradd --create-home appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY app/app.py ./app/app.py

# Copy model artifacts
COPY model/model.pkl ./model/model.pkl
COPY model/encoders.pkl ./model/encoders.pkl
COPY model/feature_config.json ./model/feature_config.json
COPY model/drift_baseline.json ./model/drift_baseline.json

# Environment variables
# MODEL_DIR: tells app.py where to find model files
# MODEL_VERSION: shows in /health and /predict responses
# PYTHONUNBUFFERED: print logs immediately (don't buffer)
ENV MODEL_DIR=/app/model
ENV MODEL_VERSION=v1
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER appuser

# Document which port the container uses
# (doesn't actually open the port — that's done at runtime)
EXPOSE 5000

# Health check: Docker-level monitoring
# Kubernetes has its own health checks, but this is defense-in-depth
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Start the application with gunicorn (production server)
# --bind: listen on all interfaces, port 5000
# --workers 2: 2 worker processes (handles concurrent requests)
# --timeout 120: kill worker if it takes > 120 seconds
# --access-logfile -: print access logs to stdout (Kubernetes collects these)
# app.app:app = file app/app.py, variable name 'app'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "app.app:app"]
