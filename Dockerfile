# ---------- Builder stage ----------
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    grep -ivE '^\s*torch\s*$' requirements.txt > requirements-notorch.txt && \
    pip install --no-cache-dir -r requirements-notorch.txt && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# ---------- Final stage ----------
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user FIRST so subsequent COPYs can use --chown
RUN addgroup --gid 1000 appuser && \
    adduser --disabled-password --uid 1000 --gid 1000 appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code with correct ownership
COPY --chown=appuser:appuser . .

# Wipe any models/ baked in from the build context and recreate it
# clean. The PVC will mount over this at runtime; perms set here
# are the fallback for when the pod runs without a volume mount.
RUN rm -rf /app/models && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app && \
    chmod -R g+rwX /app/models

USER appuser

EXPOSE 5000

ENV FLASK_APP=app.py \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

CMD ["python", "app.py"]