# Build matrix for the teff toolchain images.
#
#     docker buildx bake            # build all six locally
#     docker buildx bake --push     # build + push (used by release.yml)
#
# Overridable variables (e.g. from CI):
#     REGISTRY, NAMESPACE, VERSION

variable "REGISTRY" {
  default = "docker.io"
}

variable "NAMESPACE" {
  default = "bzdvdn"
}

variable "VERSION" {
  default = "dev"
}

group "default" {
  targets = ["core", "fastapi", "worker", "obs", "rag", "all"]
}

# CLI runner: teff run/daemon/graph/validate/... on workflow.yaml + plugins.
target "core" {
  context = "."
  args = {
    EXTRA   = "tools"
    RUNMODE = "teff"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/teff:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/teff:latest",
  ]
}

# FastAPI server base for scaffold-generated apps (uvicorn main:app).
target "fastapi" {
  context = "."
  args = {
    EXTRA   = "fastapi"
    RUNMODE = "uvicorn"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/teff-fastapi:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/teff-fastapi:latest",
  ]
}

# Celery worker/beat base for background jobs.
target "worker" {
  context = "."
  args = {
    EXTRA   = "queue"
    RUNMODE = "celery"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/teff-worker:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/teff-worker:latest",
  ]
}

# Standalone trace collector/dashboard: `teff obs-server` (ingest + UI).
target "obs" {
  context = "."
  args = {
    EXTRA   = "observability"
    RUNMODE = "obs-server"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/teff-obs:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/teff-obs:latest",
  ]
}

# Everything except `docs` (keeps mkdocs out of the image).
target "all" {
  context = "."
  args = {
    EXTRA   = "embedding,tools,rag-pdf,rag-excel,pg-checkpoint,fastapi,queue"
    RUNMODE = "teff"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/teff-all:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/teff-all:latest",
  ]
}

# Slim RAG example: `teff[embedding]` pulls in every vector store (chromadb
# alone brings onnxruntime, ~200+ MB), so build only the store you use.
#   docker buildx bake rag
target "rag" {
  context = "."
  args = {
    EXTRA   = "stores-qdrant,tools,rag-pdf"
    RUNMODE = "teff"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/teff-rag:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/teff-rag:latest",
  ]
}
