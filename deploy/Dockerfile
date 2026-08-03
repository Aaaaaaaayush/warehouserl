# ==============================================================================
# WarehouseRL — Deployment Dockerfile
# Optimized for Oracle Cloud ARM VM (Ampere A1, Ubuntu 22.04 LTS)
# ==============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install minimal server runtime dependencies (FastAPI, Uvicorn, PyYAML, Pydantic)
RUN pip install --no-cache-dir fastapi uvicorn pyyaml pydantic numpy jinja2 httpx

# Copy source code and pre-computed visualizer artifacts
COPY src/ src/
COPY configs/ configs/
COPY frontend/ frontend/
COPY logs/ logs/
COPY episodes/ episodes/
COPY videos/ videos/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/scenarios || exit 1

# Launch Uvicorn production ASGI server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
