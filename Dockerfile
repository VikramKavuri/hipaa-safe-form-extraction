# Container for the CPU-side pipeline (rendering, OCR, orchestration).
#
# NOTE: the vision-language model is served by Ollama, which runs as a
# SEPARATE process/container. Point this container at it with, e.g.:
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
#       -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" formextract \
#       formextract run --input data/sample --output outputs/run.csv
FROM python:3.11-slim

# Tesseract is the only system dependency (used for OSD + checkbox localization).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[eval]"

# tesseract is on PATH inside the image.
ENV FORMEXTRACT_TESSERACT_CMD=""

ENTRYPOINT ["formextract"]
CMD ["--help"]
