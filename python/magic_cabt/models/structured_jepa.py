"""Public surface for the structured Magic JEPA."""
from .causal import causal_delta_vector
from .jepa_model import (TORCH_AVAILABLE, MagicJEPA, StateEncoder,
                         model_parameter_count)
from .structured_config import CardTextResolver, StructuredJEPAConfig
from .structured_tensorizer import StructuredTensorizer

__all__ = [
    "TORCH_AVAILABLE", "CardTextResolver", "MagicJEPA", "StateEncoder",
    "StructuredJEPAConfig", "StructuredTensorizer", "causal_delta_vector",
    "model_parameter_count",
]
