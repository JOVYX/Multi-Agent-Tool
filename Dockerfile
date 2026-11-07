# Multi-stage production Dockerfile for Multi-Tool AI Agent
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn sse-starlette

# Final runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY pyproject.toml ./

RUN pip install --no-cache-dir -e .

# Create workspace directory with proper permissions
RUN mkdir -p /app/workspace && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: start the FastAPI web service with Uvicorn
CMD ["uvicorn", "multitool_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]

