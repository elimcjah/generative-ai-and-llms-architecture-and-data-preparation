# nlp_data_loader Dockerfile
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    TZ=UTC

WORKDIR /app

# System deps (build-essential removed after building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipenv manifests first (expect Pipfile + Pipfile.lock)
COPY Pipfile Pipfile.lock ./

# Install dependencies into system (no virtualenv)
RUN pip install --upgrade pip pipenv \
    && PIPENV_YES=1 pipenv install --system --ignore-pipfile --clear

# Download spaCy language models (pinned via latest available for spacy version)
RUN python -m spacy download en_core_web_sm && python -m spacy download de_core_news_sm

# Optionally remove build tools to slim image
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy project source (adjust if you add modules/files)
COPY . ./

# Healthcheck ensures main package import works (adjust entrypoint if added later)
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import importlib,sys; importlib.import_module('main'); sys.exit(0)" || exit 1

# Default command placeholder (replace main.py when added)
CMD ["python", "main.py"]

