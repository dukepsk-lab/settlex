"""XGBoost regression model on flat (last-step) features.

Plays the role of the report's gradient-boosting "factor" model: robust to
outliers, strong on cross-sectional fundamental/technical relationships, and
interpretable via feature importances / SHAP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Predictor


class XGBModel(Predictor):
    name = "xgboost"

    def __init__(self, feature_names: Optional[List[str]] = None, **params):
        self.feature_names = feature_names
        self.params = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=5,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=42,
        )
        self.params.update(params)
        self.model = None

    def fit(self, dataset, quick: bool = False, **kwargs) -> "XGBModel":
        from xgboost import XGBRegressor

        self.feature_names = dataset.feature_names
        params = dict(self.params)
        if quick:
            params["n_estimators"] = min(params["n_estimators"], 60)
        self.model = XGBRegressor(**params)
        self.model.fit(dataset.X_flat, dataset.y, verbose=False)
        return self

    def predict(self, X_seq: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_flat)

    def feature_importances(self):
        if self.model is None:
            return {}
        return dict(zip(self.feature_names or [], self.model.feature_importances_.tolist()))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path / "xgb.json"))
        (path / "meta.json").write_text(json.dumps({"feature_names": self.feature_names}))

    @classmethod
    def load(cls, path: Path) -> "XGBModel":
        from xgboost import XGBRegressor

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        obj = cls(feature_names=meta["feature_names"])
        obj.model = XGBRegressor()
        obj.model.load_model(str(path / "xgb.json"))
        return obj
