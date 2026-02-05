FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies in a virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    Flask==3.0.0 \
    torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    numpy==1.24.3 \
    redis==5.0.1 \
    requests==2.31.0 \
    python-dotenv==1.0.0 \
    Werkzeug==3.0.1

# Final stage
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create models directory with proper permissions
RUN mkdir -p /app/models && \
    addgroup --gid 1000 appuser && \
    adduser --disabled-password --uid 1000 --gid 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]


# # Python Flask RL Service Dockerfile
# FROM python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first (for layer caching)
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create models directory
# RUN mkdir -p /app/models

# # Expose Flask port
# EXPOSE 5000

# # Set environment variables
# ENV FLASK_APP=app.py
# ENV PYTHONUNBUFFERED=1

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# # Run the application
# CMD ["python", "app.py"]