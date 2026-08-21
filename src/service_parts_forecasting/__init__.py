"""Global service-parts forecasting package."""

import os

# CUDA deterministic matrix multiplication requires this to be set before the
# first cuBLAS handle is created. ``setdefault`` preserves an explicit user choice.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "0.1.0"
