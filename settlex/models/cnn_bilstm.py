"""CNN-BiLSTM sequence regressor (the report's recommended architecture).

Conv1D layers extract local temporal / cross-feature patterns, which are then
fed to stacked Bidirectional LSTMs that capture sequential memory in both
directions. Dropout + L2 + early stopping provide the aggressive regularisation
the report calls for given the small SET50 universe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Predictor


class CNNBiLSTMModel(Predictor):
    name = "cnn_bilstm"

    def __init__(
        self,
        lookback: int = 60,
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        conv_filters: int = 32,
        lstm_units: int = 32,
        dropout: float = 0.3,
        l2: float = 1e-4,
    ):
        self.lookback = lookback
        self.n_features = n_features
        self.feature_names = feature_names
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.l2 = l2
        self.model = None

    def _build(self, n_features: int):
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        reg = regularizers.l2(self.l2)
        model = keras.Sequential(
            [
                layers.Input(shape=(self.lookback, n_features)),
                layers.Conv1D(self.conv_filters, 3, padding="causal", activation="relu", kernel_regularizer=reg),
                layers.BatchNormalization(),
                layers.Conv1D(self.conv_filters, 3, padding="causal", activation="relu", kernel_regularizer=reg),
                layers.Dropout(self.dropout),
                layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True, kernel_regularizer=reg)),
                layers.Dropout(self.dropout),
                layers.Bidirectional(layers.LSTM(self.lstm_units, kernel_regularizer=reg)),
                layers.Dropout(self.dropout),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
        return model

    def fit(
        self,
        dataset,
        epochs: int = 40,
        batch_size: int = 64,
        validation_split: float = 0.15,
        verbose: int = 0,
        quick: bool = False,
        seed: int = 42,
    ) -> "CNNBiLSTMModel":
        import tensorflow as tf
        from tensorflow import keras

        tf.keras.utils.set_random_seed(seed)
        self.feature_names = dataset.feature_names
        self.lookback = int(dataset.X_seq.shape[1])
        self.n_features = int(dataset.X_seq.shape[2])
        self.model = self._build(self.n_features)
        if quick:
            epochs = min(epochs, 3)
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=6, restore_best_weights=True, monitor="val_loss"
            )
        ]
        self.model.fit(
            dataset.X_seq,
            dataset.y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose,
        )
        return self

    def predict(self, X_seq: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        return self.model.predict(X_seq, verbose=0).ravel()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path / "cnn_bilstm.keras"))
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "lookback": self.lookback,
                    "n_features": self.n_features,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "CNNBiLSTMModel":
        from tensorflow import keras

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        obj = cls(
            lookback=meta["lookback"],
            n_features=meta["n_features"],
            feature_names=meta["feature_names"],
        )
        obj.model = keras.models.load_model(str(path / "cnn_bilstm.keras"))
        return obj
