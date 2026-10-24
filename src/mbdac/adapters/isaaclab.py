"""Isaac Lab bridge; SDK launch remains owned by the user's training entry point.

Accept the official RslRlVecEnvWrapper plus task-provided pre-reset snapshots.
This module intentionally imports neither Isaac Lab nor MJLab.
"""
from .base import RslModelAdapter


class IsaacLabModelAdapter(RslModelAdapter):
    pass
