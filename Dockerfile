# ---------- Builder stage ----------
FROM python:3.10-slim AS builder

# Build deps (some python libs may need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
# 1) Install everything except torch from PyPI
# 2) Install torch CPU wheels from PyTorch index
RUN pip install --no-cache-dir --upgrade pip && \
    grep -ivE '^\s*torch\s*$' requirements.txt > requirements-notorch.txt && \
    pip install --no-cache-dir -r requirements-notorch.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# ---------- Final stage ----------
FROM python:3.10-slim

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user FIRST so subsequent COPYs can use --chown
RUN addgroup --gid 1000 appuser && \
    adduser --disabled-password --uid 1000 --gid 1000 appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code with correct ownership from the start
COPY --chown=appuser:appuser . .

# Create models directory with correct ownership and permissive mode.
# Mode 0775 (group-writable) helps when a host volume is mounted over it
# with a different UID but a matching GID — common on k8s with fsGroup.
RUN mkdir -p /app/models && \
    chown -R appuser:appuser /app/models && \
    chmod -R 0775 /app/models

# Declare /app/models as a volume so it's clearly a writable mount point
VOLUME ["/app/models"]

USER appuser

EXPOSE 5000

ENV FLASK_APP=app.py \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

CMD ["python", "app.py"]