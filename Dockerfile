# Hakam: FastAPI backend + built React frontend in one image.
#
#   docker build -t hakam .
#   docker run -p 7860:7860 --env-file .env -e HAKAM_WEIGHTS_REPO=user/hakam-weights -e HF_TOKEN=hf_... hakam
#
# Works as a Hugging Face Docker Space (port 7860, runs as uid 1000) and on any Docker host.

# ---- 1. frontend (skipped until web/package.json exists) ------------------
FROM node:20-slim AS web
WORKDIR /repo
COPY . .
RUN if [ -f web/package.json ]; then cd web && npm ci && npm run build; fi && mkdir -p web/dist

# ---- 2. backend -------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-deploy.txt

COPY . .
COPY --from=web /repo/web/dist ./web/dist

# Download the embedding model and the VideoMAE architecture at build time, so the
# first request after a restart is not a 1 GB download.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')" \
    && python -c "from transformers import VideoMAEConfig; VideoMAEConfig.from_pretrained('MCG-NJU/videomae-base-finetuned-kinetics')"

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
