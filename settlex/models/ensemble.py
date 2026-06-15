"""Ensemble that averages CNN-BiLSTM and XGBoost predictions.

A simple Mixture-of-Experts style weighted average of the two members'
predicted forward returns. Both members predict the same target (forward
``horizon``-day return), so averaging in return-space is meaningful and
mitigates single-model bias, as recommended in the report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Predictor
from .cnn_bilstm import CNNBiLSTMModel
from .xgb import XGBModel

_MEMBER_TYPES = {
    "cnn_bilstm": CNNBiLSTMModel,
    "xgboost": XGBModel,
}


class Ensemble(Predictor):
    name = "ensemble"

    def __init__(self, members: Optional[List[Predictor]] = None, weights: Optional[List[float]] = None):
        self.members = members or []
        self.weights = weights

    def fit(self, dataset, quick: bool = False, verbose: int = 0, **kwargs) -> "Ensemble":
        self.feature_names = dataset.feature_names
        self.members = [
            CNNBiLSTMModel(lookback=int(dataset.X_seq.shape[1])),
            XGBModel(),
        ]
        for member in self.members:
            member.fit(dataset, quick=quick, verbose=verbose)
        if self.weights is None:
            self.weights = [1.0 / len(self.members)] * len(self.members)
        return self

    def predict(self, X_seq: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        if not self.members:
            raise RuntimeError("Ensemble has no members; train or load it first.")
        preds = np.vstack([np.asarray(m.predict(X_seq, X_flat), dtype=float) for m in self.members])
        weights = np.asarray(self.weights, dtype=float).reshape(-1, 1)
        return (preds * weights).sum(axis=0)

    def member_predictions(self, X_seq: np.ndarray, X_flat: np.ndarray):
        return {m.name: np.asarray(m.predict(X_seq, X_flat), dtype=float) for m in self.members}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        member_meta = []
        for i, member in enumerate(self.members):
            sub = f"member_{i}_{member.name}"
            member.save(path / sub)
            member_meta.append({"name": member.name, "dir": sub})
        (path / "ensemble.json").write_text(
            json.dumps(
                {
                    "members": member_meta,
                    "weights": self.weights,
                    "feature_names": self.feature_names,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "Ensemble":
        path = Path(path)
        meta_file = path / "ensemble.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"No trained ensemble at {path}. Run `settlex train` first."
            )
        meta = json.loads(meta_file.read_text())
        members = []
        for member_meta in meta["members"]:
            model_cls = _MEMBER_TYPES[member_meta["name"]]
            members.append(model_cls.load(path / member_meta["dir"]))
        obj = cls(members=members, weights=meta["weights"])
        obj.feature_names = meta.get("feature_names")
        return obj
