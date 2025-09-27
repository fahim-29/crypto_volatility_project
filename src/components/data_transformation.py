"""Build and apply preprocessing pipeline.
Contains functions to build ColumnTransformer, save transformer, and transform DataFrames.
"""
from pathlib import Path
import joblib
import pandas as pd
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from ..utils.utils import create_dir
from ..utils.logger import get_logger


logger = get_logger(__name__)




class DataTransformation:
def __init__(self, artifacts_dir: str = "artifacts"):
self.artifacts_dir = Path(artifacts_dir)
self.transformer_dir = self.artifacts_dir / "transformer"
create_dir(self.transformer_dir)


def build_preprocessor(self, numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
# numerical pipeline
num_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
])


# categorical pipeline
cat_pipeline = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])


preprocessor = ColumnTransformer(transformers=[
("num", num_pipeline, numerical_cols),
("cat", cat_pipeline, categorical_cols)
])


logger.info("Built preprocessor ColumnTransformer")
return preprocessor


def save_preprocessor(self, preprocessor, filename: str = "preprocessor.joblib") -> Path:
out_path = self.transformer_dir / filename
joblib.dump(preprocessor, out_path)
logger.info(f"Saved preprocessor to: {out_path}")
return out_path


def load_preprocessor(self, filepath: str):
return joblib.load(filepath)


def transform(self, preprocessor, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
"""Apply preprocessor to df and return X, y (assumes target column exists)
Note: this returns numpy arrays from transformer; caller should handle column names.
"""
# expected: caller provides columns
try:
X = preprocessor.transform(df)
logger.info("Applied preprocessor to DataFrame")
return X
except Exception as e:
logger.exception("Failed to transform data")
raise