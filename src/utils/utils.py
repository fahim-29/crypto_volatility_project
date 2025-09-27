from pathlib import Path
import joblib
import json
import os
from typing import Dict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np




def create_dir(path: Path):
Path(path).mkdir(parents=True, exist_ok=True)




def save_pickle(obj, path: str):
joblib.dump(obj, path)




def load_pickle(path: str):
return joblib.load(path)




def evaluate(y_true, y_pred) -> Dict[str, float]:
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true