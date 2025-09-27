

"""
Model Trainer
-------------
This module trains and evaluates models, and saves the best pipeline.
"""

from pathlib import Path
import joblib
import numpy as np
from typing import Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import helpers
from src.utils.utils import create_dir
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_dir = self.artifacts_dir / "models"
        create_dir(self.model_dir)

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """Evaluate predictions with RMSE, MAE, and R²."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"rmse": rmse, "mae": mae, "r2": r2}

    def save_model(self, pipeline, filename: str = "best_model.joblib") -> Path:
        """Save trained pipeline to artifacts/models/"""
        out_path = self.model_dir / filename
        joblib.dump(pipeline, out_path)
        logger.info(f"✅ Saved model pipeline to: {out_path}")
        return out_path

    def load_model(self, filepath: str):
        """Load trained pipeline from file"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # 🔹 Simple test block
    trainer = ModelTrainer()
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    metrics = trainer.evaluate(y_true, y_pred)
    print("Test evaluation:", metrics)
