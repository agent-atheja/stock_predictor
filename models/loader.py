"""Load the persisted ensemble model for serving / signal generation / paper trading."""
from __future__ import annotations

from core.config import load_config, resolve


def load_final_model():
    """Load the CrossSectionalModel saved by training (joblib — supports mixed LGB/XGB members)."""
    import joblib

    reg = resolve(load_config().model.registry_dir) / "final_model"
    return joblib.load(reg / "model.joblib")
