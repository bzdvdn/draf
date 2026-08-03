# Build matrix for the draf toolchain images.
#
#     docker buildx bake            # build all four locally
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
  targets = ["core", "fastapi", "worker", "all"]
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
    EXTRA   = "embedding,tools,mcp,rag-pdf,rag-excel,pg-checkpoint,fastapi,queue"
    RUNMODE = "draf"
  }
  tags = [
    "${REGISTRY}/${NAMESPACE}/draf-all:${VERSION}",
    "${REGISTRY}/${NAMESPACE}/draf-all:latest",
  ]
}
