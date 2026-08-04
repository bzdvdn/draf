# Build matrix for the draf toolchain images.
#
#     docker buildx bake            # build all five locally
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
  targets = ["core", "fastapi", "worker", "rag", "all"]
}

# CLI runner: draf run/daemon/graph/validate/... on workflow.yaml + plugins.
target "core" {
  context = "."
  args = {
    EXTRA   = "tools"
    RUNMODE = "draf"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/draf:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/draf:latest",
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
    "${REGISTRY}/${NAMESPACE}/draf-fastapi:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/draf-fastapi:latest",
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
    "${REGISTRY}/${NAMESPACE}/draf-worker:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/draf-worker:latest",
  ]
}

# Everything except `docs` (keeps mkdocs out of the image).
target "all" {
  context = "."
  args = {
    EXTRA   = "embedding,tools,rag-pdf,rag-excel,pg-checkpoint,fastapi,queue"
    RUNMODE = "draf"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/draf-all:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/draf-all:latest",
  ]
}

# Slim RAG example: `draf[embedding]` pulls in every vector store (chromadb
# alone brings onnxruntime, ~200+ MB), so build only the store you use.
#   docker buildx bake rag
target "rag" {
  context = "."
  args = {
    EXTRA   = "stores-qdrant,tools,rag-pdf"
    RUNMODE = "draf"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/draf-rag:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/draf-rag:latest",
  ]
}
