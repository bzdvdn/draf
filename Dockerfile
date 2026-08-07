# syntax=docker/dockerfile:1
# teff toolchain images — one build, five variants.
#
# Variant           EXTRA                             RUNMODE
# teff (core)       tools                             teff
# teff-fastapi      fastapi                           uvicorn
# teff-worker       queue                             celery
# teff-obs          observability                     obs-server
# teff-rag          stores-qdrant,tools,rag-pdf       teff
# teff-all          embedding,tools,rag-pdf,          teff
#                   rag-excel,pg-checkpoint,fastapi,queue
#
# `embedding` is an alias for every vector store; for a slim RAG image
# install only the store you use, e.g.:
#     docker build --build-arg EXTRA=stores-qdrant,tools -t teff-rag .
#
# Build individually:
#     docker build --build-arg EXTRA=tools --build-arg RUNMODE=teff -t teff .
# or the whole matrix at once:
#     docker buildx bake

ARG EXTRA=tools
ARG RUNMODE=teff

# ------------------------------------------------- build the dependencies ----
FROM python:3.12-slim AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

# Only what `pip wheel` needs to resolve/isolate the project.
COPY pyproject.toml README.md ./
COPY teff ./teff

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
RUN pip install --no-index --find-links=/wheels "teff[${EXTRA}]" \
    && rm -rf /wheels \
    && mkdir -p /workflow /data/checkpoints \
    && chown -R 65534:65534 /workflow /data

# Durable runtime directory (checkpoints).  Declared and owned here so a
# mounted named volume inherits this ownership on first use.
VOLUME ["/data/checkpoints"]

# Non-root user (65534 = nobody).
USER 65534

# The variant binary is selected at build time (teff | uvicorn | celery |
# obs-server); pass any remaining arguments on the command line, e.g.
#   docker run teff run -f /workflow/workflow.yaml
#   docker run teff-fastapi main:app --host 0.0.0.0
#   docker run teff-worker -A src.celery_app worker
#   docker run teff-obs --db /data/traces.db --host 0.0.0.0 --port 8001
ENTRYPOINT ["/bin/sh", "-c", "case \"$RUNMODE\" in obs-server) exec teff obs-server \"$@\" ;; *) exec \"$RUNMODE\" \"$@\" ;; esac", "--"]
CMD ["--help"]
