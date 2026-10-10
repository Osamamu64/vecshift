"""VecShift: safe, observable embedding migrations for any vector store."""

from vecshift.core.capabilities import Capability
from vecshift.core.fingerprint import EmbeddingFingerprint
from vecshift.core.record import Record, SparseVector

__version__ = "0.3.0"

__all__ = [
    "Capability",
    "EmbeddingFingerprint",
    "Record",
    "SparseVector",
    "__version__",
]
