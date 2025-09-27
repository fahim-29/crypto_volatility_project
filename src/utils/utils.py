



from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict


def create_dir(path: Path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path: str):
    """Save Python object as a pickle file."""
    joblib.dump(obj, path)


def load_pickle(path: str):
    """Load Python object from a pickle file."""
    return joblib.load(path)


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Evaluate predictions with RMSE, MAE, and R²."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}












