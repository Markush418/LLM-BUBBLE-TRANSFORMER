"""
Riemannian Optimizer Utilities for Bubble Transformer V4.

Provides utilities for creating optimizers that handle ManifoldParameters
with appropriate Riemannian optimization algorithms.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union

try:
    import geoopt

    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    geoopt = None


def get_manifold_parameters(module: nn.Module) -> List[torch.nn.Parameter]:
    """
    Extract ManifoldParameters from a module.

    Scans all parameters in the module and returns those that are
    ManifoldParameter instances (parameters with manifold attribute).

    Args:
        module: PyTorch module to scan

    Returns:
        List of ManifoldParameter instances found in the module
    """
    manifold_params = []
    for param in module.parameters():
        # Check if parameter has manifold attribute or is ManifoldParameter
        if hasattr(param, "manifold") or (
            GEOOPT_AVAILABLE and isinstance(param, geoopt.ManifoldParameter)
        ):
            manifold_params.append(param)
    return manifold_params


def get_regular_parameters(module: nn.Module) -> List[torch.nn.Parameter]:
    """
    Extract regular (non-manifold) parameters from a module.

    Args:
        module: PyTorch module to scan

    Returns:
        List of regular Parameter instances
    """
    manifold_params = get_manifold_parameters(module)
    manifold_set = set(id(p) for p in manifold_params)
    return [p for p in module.parameters() if id(p) not in manifold_set]


def create_riemannian_optimizer(
    module: nn.Module,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer with RiemannianAdam for ManifoldParameters.

    Automatically separates ManifoldParameters (which need Riemannian optimization)
    from regular parameters (which use standard optimization).

    Args:
        module: Module containing parameters to optimize
        lr: Learning rate (default: 1e-3)
        optimizer_type: 'adam' or 'sgd' (default: 'adam')
        weight_decay: Weight decay coefficient (default: 0.0)
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer (RiemannianAdam if geoopt available, else Adam)

    Example:
        >>> from models.bubble_centroids_v4 import BubbleCentroidsV4
        >>> module = BubbleCentroidsV4(8, 32, 64, manifold_type='poincare')
        >>> optimizer = create_riemannian_optimizer(module, lr=1e-3)
    """
    # Fallback to standard Adam if geoopt not available
    if not GEOOPT_AVAILABLE:
        return torch.optim.Adam(
            module.parameters(), lr=lr, weight_decay=weight_decay, **kwargs
        )

    # Separate manifold and regular parameters
    manifold_params = get_manifold_parameters(module)
    regular_params = get_regular_parameters(module)

    # Select optimizer class based on type
    if optimizer_type == "adam":
        manifold_optimizer_class = geoopt.optim.RiemannianAdam
        regular_optimizer_class = torch.optim.Adam
    elif optimizer_type == "sgd":
        manifold_optimizer_class = geoopt.optim.RiemannianSGD
        regular_optimizer_class = torch.optim.SGD
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'sgd'."
        )

    # Case 1: Only manifold parameters
    if manifold_params and not regular_params:
        return manifold_optimizer_class(
            manifold_params, lr=lr, weight_decay=weight_decay, **kwargs
        )

    # Case 2: Only regular parameters
    if regular_params and not manifold_params:
        return regular_optimizer_class(
            regular_params, lr=lr, weight_decay=weight_decay, **kwargs
        )

    # Case 3: Mixed parameters - use RiemannianAdam for all
    # This is simpler and geoopt handles regular params correctly
    if manifold_params and regular_params:
        # Combine all parameters
        all_params = list(module.parameters())
        return manifold_optimizer_class(
            all_params, lr=lr, weight_decay=weight_decay, **kwargs
        )

    # Edge case: no parameters
    return torch.optim.Adam(module.parameters(), lr=lr)


def create_optimizer_with_separate_groups(
    module: nn.Module,
    lr_manifold: float = 1e-3,
    lr_regular: float = 1e-3,
    optimizer_type: str = "adam",
    weight_decay_manifold: float = 0.0,
    weight_decay_regular: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer with separate learning rates for manifold and regular parameters.

    This provides fine-grained control over optimization of different parameter types.

    Args:
        module: Module containing parameters to optimize
        lr_manifold: Learning rate for manifold parameters (default: 1e-3)
        lr_regular: Learning rate for regular parameters (default: 1e-3)
        optimizer_type: 'adam' or 'sgd' (default: 'adam')
        weight_decay_manifold: Weight decay for manifold params (default: 0.0)
        weight_decay_regular: Weight decay for regular params (default: 0.0)
        **kwargs: Additional optimizer arguments

    Returns:
        Configured optimizer with separate parameter groups
    """
    if not GEOOPT_AVAILABLE:
        return torch.optim.Adam(
            module.parameters(),
            lr=lr_regular,
            weight_decay=weight_decay_regular,
            **kwargs,
        )

    manifold_params = get_manifold_parameters(module)
    regular_params = get_regular_parameters(module)

    # Select optimizer class
    if optimizer_type == "adam":
        manifold_optimizer_class = geoopt.optim.RiemannianAdam
    elif optimizer_type == "sgd":
        manifold_optimizer_class = geoopt.optim.RiemannianSGD
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Build parameter groups
    param_groups = []

    if manifold_params:
        param_groups.append(
            {
                "params": manifold_params,
                "lr": lr_manifold,
                "weight_decay": weight_decay_manifold,
            }
        )

    if regular_params:
        param_groups.append(
            {
                "params": regular_params,
                "lr": lr_regular,
                "weight_decay": weight_decay_regular,
            }
        )

    if not param_groups:
        # No parameters
        return torch.optim.Adam(module.parameters(), lr=lr_regular)

    return manifold_optimizer_class(param_groups, **kwargs)


# Convenience exports
__all__ = [
    "get_manifold_parameters",
    "get_regular_parameters",
    "create_riemannian_optimizer",
    "create_optimizer_with_separate_groups",
    "GEOOPT_AVAILABLE",
]
