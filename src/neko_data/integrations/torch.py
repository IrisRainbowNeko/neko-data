"""Optional, explicit CPU tensor conversion for the safe typed codec."""

from __future__ import annotations

from ..contract.typed import TensorValue


def from_torch_tree(value):
    """Detach/copy torch tensors while preserving containers and other leaves."""
    import torch

    if isinstance(value, torch.Tensor):
        return TensorValue.from_torch(value)
    if isinstance(value, dict):
        return {key: from_torch_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [from_torch_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(from_torch_tree(child) for child in value)
    return value


def to_torch_tree(value):
    """Restore TensorValue leaves as fresh CPU torch tensors; NumPy stays NumPy."""
    if isinstance(value, TensorValue):
        return value.to_torch()
    if isinstance(value, dict):
        return {key: to_torch_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [to_torch_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(to_torch_tree(child) for child in value)
    return value
