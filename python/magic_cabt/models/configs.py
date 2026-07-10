"""Model-size configurations for first action-space ablations.

The configs are framework-neutral. A later PyTorch/JAX trainer can consume the
same values, while stdlib tests can still validate intended capacity.
"""

__all__ = ["MODEL_CONFIGS", "get_model_config", "estimate_parameter_count"]

MODEL_CONFIGS = {
    "small": {
        "name": "small",
        "description": "Coarse-action model for quick BC/PPO smoke tests.",
        "stateFeatureDim": 256,
        "optionFeatureDim": 128,
        "hiddenDim": 256,
        "layers": 2,
        "actionProfile": "small",
        "auxiliaryFactors": False,
    },
    "full": {
        "name": "full",
        "description": "Larger legal-option model with causal auxiliary outputs.",
        "stateFeatureDim": 512,
        "optionFeatureDim": 256,
        "hiddenDim": 768,
        "layers": 4,
        "actionProfile": "full",
        "auxiliaryFactors": True,
        "auxiliaryFactorCount": 18,
    },
}


def get_model_config(name):
    """Return a copy of a named model config."""
    if name not in MODEL_CONFIGS:
        raise ValueError("unknown model config: %r" % (name,))
    return dict(MODEL_CONFIGS[name])


def estimate_parameter_count(config):
    """Return a rough MLP option-ranker parameter estimate."""
    hidden = int(config.get("hiddenDim", 0))
    state_dim = int(config.get("stateFeatureDim", 0))
    option_dim = int(config.get("optionFeatureDim", 0))
    layers = int(config.get("layers", 0))
    params = 0
    params += (state_dim * hidden) + hidden
    params += (option_dim * hidden) + hidden
    for _ in range(max(0, layers)):
        params += (hidden * hidden) + hidden
    params += hidden + 1
    if config.get("auxiliaryFactors"):
        count = int(config.get("auxiliaryFactorCount", 0))
        params += count * (hidden + 1)
    return params
