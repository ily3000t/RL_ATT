"""Observation attacks independent of victim training."""

from .base import BaseAttack, AttackContext, AttackResult
from .registry import AttackRegistry

__all__ = ["BaseAttack", "AttackContext", "AttackResult", "AttackRegistry"]
