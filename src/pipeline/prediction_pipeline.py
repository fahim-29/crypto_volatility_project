"""Simple prediction utility that loads saved pipeline and predicts on new data.
Example usage:
from src.pipeline.prediction_pipeline import load_pipeline_and_predict
"""
from pathlib import Path
import joblib
import pandas as pd
from typing import Any
from ..utils.logger import get_logger


logger = get_logger(__name__)




def load_pipeline(pipeline_path: str):
return joblib.load(pipeline_path)




def predict(pipeline, df: pd.DataFrame) -> Any:
"""Returns predictions for incoming dataframe (df must include same features as training X)
"""
preds = pipeline.predict(df)
return preds