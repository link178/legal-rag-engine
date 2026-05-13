"""Generation pipeline errors (no SQLAlchemy imports)."""


class GenerationError(Exception):
    """Base class for grounded generation failures."""


class UnsupportedProviderError(GenerationError):
    """Raised when an unsupported generation provider is requested."""
