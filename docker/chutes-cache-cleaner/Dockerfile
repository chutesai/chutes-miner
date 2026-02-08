# Cache cleanup image: HF/CivitAI cache cleanup. Script is in the image (no ConfigMap).
# Runs as UID 1000; host cache dir must be owned by 1000:1000 when VM is set up.
FROM python:3.12-slim AS base
ARG PROJECT_DIR
ARG PROJECT

RUN pip install --no-cache-dir huggingface_hub

WORKDIR /app
COPY ${PROJECT_DIR} ./
RUN pip install --no-cache-dir .

# Run as root so init can create/chown cache dirs when kubelet creates hostPath as root.
# Chute init: mkdir then run cleanup. Standalone: just run cleanup (mkdir is harmless).
ENTRYPOINT ["sh", "-c", "mkdir -p /cache/hub /cache/civitai && chmod -R 777 /cache && exec python -m chutes_cache_cleaner.cache_cleanup"]

FROM base AS production

FROM base AS development
