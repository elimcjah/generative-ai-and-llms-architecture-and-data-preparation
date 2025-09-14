# Use official Python slim image
FROM python:3.11-slim AS runtime

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HEARTBEAT_INTERVAL=5 \
    TZ=UTC

WORKDIR /app

# Install minimal system deps (adjust if TF/Torch/TensorFlow need extras)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipenv manifests first for caching
COPY Pipfile Pipfile.lock ./

# Install dependencies via pipenv into system site-packages (no venv)
RUN pip install --upgrade pip pipenv \
    && PIPENV_YES=1 pipenv install --system --ignore-pipfile --clear

# Copy source
COPY chatbot/ ./chatbot/

# Runtime command
CMD ["python", "chatbot/main.py"]

# Healthcheck: lightweight import test
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import importlib,sys; importlib.import_module('chatbot.main'); sys.exit(0)" || exit 1
