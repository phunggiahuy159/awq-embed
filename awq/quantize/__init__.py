try:
    from .w8a8_linear import *
except ImportError as exc:
    if "awq_inference_engine" not in str(exc):
        raise
from .smooth import *
