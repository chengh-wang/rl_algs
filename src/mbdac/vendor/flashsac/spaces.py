"""Minimal immutable shape/dtype descriptor; replay never uses space sampling."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ShapeSpace:
    shape: tuple[int, ...]
    dtype: object = np.float32
