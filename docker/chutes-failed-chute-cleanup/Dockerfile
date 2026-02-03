# Failed chute cleanup: delete Failed pods and orphaned chute Services. Uses kubectl; entrypoint from image.
FROM bitnamilegacy/kubectl:latest AS base

WORKDIR /app
COPY --chmod=755 docker/chutes-failed-chute-cleanup/cleanup.sh /app/cleanup.sh

ENTRYPOINT ["/app/cleanup.sh"]

FROM base AS production

FROM base AS development
