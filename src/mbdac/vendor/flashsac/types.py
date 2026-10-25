"""PyTorch/NumPy-only annotations; upstream JAX dependency is unused here."""
from typing import Any, Union
import numpy.typing as npt
import torch
NDArray = npt.NDArray[Any]
Tensor = Union[NDArray, torch.Tensor]
