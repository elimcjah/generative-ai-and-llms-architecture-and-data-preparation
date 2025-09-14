# Use official Python slim image
FROM python:3.11-slim AS runtime

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HEARTBEAT_INTERVAL=5 \
    TZ=UTC

WORKDIR /app

# Install minimal system deps plus build tools (removed later) for compiling wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipenv manifests first for caching
COPY Pipfile Pipfile.lock ./

# Install dependencies via pipenv into system site-packages (no venv)
RUN pip install --upgrade pip pipenv \
    && PIPENV_YES=1 pipenv install --system --ignore-pipfile --clear

# Download spaCy language models
RUN python -m spacy download en_core_web_sm && python -m spacy download de_core_news_sm

# Optionally remove build toolchain to slim image (keep runtime libs)
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy source (create tokenization/ directory and place main.py inside it)
COPY main.py tokenization/main.py

# Runtime command
CMD ["python", "tokenization/main.py"]

# Healthcheck: lightweight import test (NOTE: adjust module name if needed)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import importlib,sys; importlib.import_module('tokenization.main'); sys.exit(0)" || exit 1
