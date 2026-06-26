from .base import MemoryBackend, MemoryRecord
from .vector_baseline import VectorBaselineBackend

__all__ = ["MemoryBackend", "MemoryRecord", "VectorBaselineBackend"]

try:
    from .mem0_adapter import Mem0Backend
    __all__.append("Mem0Backend")
except ImportError:
    pass
