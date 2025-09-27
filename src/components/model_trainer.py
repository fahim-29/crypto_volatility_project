"""Train models and save the best model.
This module trains a set of models, evaluates them, and persists the best pipeline.
"""
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict
from ..utils.utils import create_dir
from ..utils.logger import get_logger


logger = get_logger(__name__)




class ModelTrainer:
def __init__(self, artifacts_dir: str = "artifacts"):
self.artifacts_dir = Path(artifacts_dir)
self.model_dir = self.artifacts_dir / "models"
create_dir(self.model_dir)


def evaluate(self, y_true, y_pred) -> Dict[str, float]:
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
return {"rmse": rmse, "mae": mae, "r2": r2}


def save_model(self, pipeline, filename: str = "best_model.joblib") -> Path:
out_path = self.model_dir / filename
joblib.dump(pipeline, out_path)
logger.info(f"Saved model pipeline to: {out_path}")
return out_path


def load_model(self, filepath: str):
return joblib.load(filepath)