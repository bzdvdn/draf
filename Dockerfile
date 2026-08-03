# syntax=docker/dockerfile:1
# draf toolchain images — one build, four variants.
#
# Variant           EXTRA                             RUNMODE
# draf (core)       tools                             draf
# draf-fastapi      fastapi                           uvicorn
# draf-worker       queue                             celery
# draf-all          embedding,tools,mcp,rag-pdf,      draf
#                   rag-excel,pg-checkpoint,fastapi,queue
#
# Build individually:
#     docker build --build-arg EXTRA=tools --build-arg RUNMODE=draf -t draf .
# or the whole matrix at once:
#     docker buildx bake

ARG EXTRA=tools
ARG RUNMODE=draf

# ------------------------------------------------- build the dependencies ----
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

# Only what `pip wheel` needs to resolve/isolate the project.
COPY pyproject.toml README.md ./
COPY draf ./draf

ARG EXTRA
RUN pip install --upgrade pip \
    && pip wheel -w /wheels ".[${EXTRA}]"

# ------------------------------------------------------------- runtime ------
FROM python:3.12-slim AS runtime
ARG EXTRA
ARG RUNMODE
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    RUNMODE=${RUNMODE}

WORKDIR /app

# Install the wheels built above; drop the wheel cache.
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels "draf[${EXTRA}]" \
    && rm -rf /wheels \
    && mkdir -p /workflow /data/checkpoints \
    && chown -R 65534:65534 /workflow /data

# Durable runtime directory (checkpoints).  Declared and owned here so a
# mounted named volume inherits this ownership on first use.
VOLUME ["/data/checkpoints"]

# Non-root user (65534 = nobody).
USER 65534

# The variant binary is selected at build time (draf | uvicorn | celery);
# pass any remaining arguments on the command line, e.g.
#   docker run draf run -f /workflow/workflow.yaml
#   docker run draf-fastapi main:app --host 0.0.0.0
#   docker run draf-worker -A src.celery_app worker
ENTRYPOINT ["/bin/sh", "-c", "exec \"$RUNMODE\" \"$@\"", "--"]
CMD ["--help"]
