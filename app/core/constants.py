"""Shared constants (service name, retrieval modes for config validation)."""

SERVICE_NAME = "legal-rag-engine"
# Backwards compatibility with early scaffold
PROJECT_NAME = SERVICE_NAME

DEFAULT_RETRIEVAL_MODE = "hybrid"
SUPPORTED_RETRIEVAL_MODES = ("dense_only", "sparse_only", "hybrid")
