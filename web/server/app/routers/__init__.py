"""
HTTP routers exposed by the FastAPI app.

- model_router    : adaptive Keras / TFLite downloads
- metrics_router  : per-round FL metrics + chart
- fl_process_router : Flower subprocess status / ensure / restart
"""

from .fl_process_router import fl_process_router
from .metrics_router import metrics_router
from .model_router import model_router

__all__ = ["metrics_router", "model_router", "fl_process_router"]
