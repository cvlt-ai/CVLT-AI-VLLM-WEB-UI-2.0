# CVLT AI vLLM Gradio WebUI 2.0 - Dockerfile
# Multi-stage build for optimized image size

# ==============================================================================
# Stage 1: Builder - Install dependencies
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip wheel setuptools

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ==============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory and set as working directory
WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set permissions
RUN chmod -R 755 /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command
CMD ["python", "main.py"]
