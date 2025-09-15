# nlp_data_loader Dockerfile (Pipenv-based)
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    TZ=UTC \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies: build-essential for any source builds (wheels preferred), libgomp1 for OpenMP used by PyTorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libjpeg62-turbo \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Pipfile (lock intentionally omitted until regenerated for these pins)
COPY Pipfile ./

# Install dependencies from Pipfile without lock (pins are in Pipfile)
RUN pip install --upgrade pip pipenv \
    && PIPENV_YES=1 pipenv install --system --skip-lock --clear

# Download spaCy language models (fail fast if unavailable)
RUN python -m spacy download en_core_web_sm && python -m spacy download de_core_news_sm

# Remove build toolchain to slim final image (keep runtime libs like libgomp1)
RUN apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy project source
COPY . ./

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import importlib,sys; importlib.import_module('main'); sys.exit(0)" || exit 1

CMD ["python", "main.py"]
