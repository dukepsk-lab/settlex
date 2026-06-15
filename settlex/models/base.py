"""Common predictor interface.

Every model takes a :class:`~settlex.features.pipeline.Dataset` to ``fit`` and,
at inference, both the sequence tensor and the flat matrix (each model uses
whichever it needs) so the ensemble can call them uniformly.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Optional

import numpy as np


class Predictor(abc.ABC):
    name: str = "base"

    feature_names: Optional[List[str]] = None

    @abc.abstractmethod
    def fit(self, dataset, **kwargs) -> "Predictor":
        ...

    @abc.abstractmethod
    def predict(self, X_seq: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        ...

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "Predictor":
        ...
