# syntax=docker/dockerfile:1

# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is small (~150MB) and has no unnecessary system packages.
FROM python:3.11-slim

# ── Environment variables ──────────────────────────────────────────────────────
# PYTHONDONTWRITEBYTECODE: don't create .pyc files (saves space in the container)
# PYTHONUNBUFFERED:        print logs immediately instead of buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# ── System dependencies ───────────────────────────────────────────────────────
# We install libgomp for scikit-learn (needed for parallel Random Forest).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies FIRST (layer caching) ─────────────────────────
# WHY copy requirements first:
#   Docker caches each layer. If we copy requirements.txt first and then
#   run pip install, Docker can reuse the "install" layer on the next build
#   as long as requirements.txt didn't change (even if app code changed).
#   This makes rebuilds MUCH faster during development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY app/ app/
COPY models/ models/
COPY src/ src/

# ── Create non-root user (security best practice) ─────────────────────────────
# WHY non-root: if the container is ever compromised, the attacker only has
# limited permissions rather than full root access.
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will call GET /health every 30s. If it fails 3 times the container
# is marked unhealthy and container orchestrators (ECS, App Runner) restart it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# ── Start command ──────────────────────────────────────────────────────────────
# Render sets $PORT automatically. We use ${PORT:-8000} as a fallback.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
