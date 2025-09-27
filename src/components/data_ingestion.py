
"""Data ingestion component.
Loads raw data (CSV / parquet / API) and saves a copy to artifacts/raw/.
"""
from pathlib import Path
import pandas as pd
from typing import Optional
from ..utils.utils import create_dir
from ..utils.logger import get_logger
from ..utils.exception import CustomException


logger = get_logger(__name__)




class DataIngestion:
def __init__(self, data_path: str, artifacts_dir: str = "artifacts"):
self.data_path = Path(data_path)
self.artifacts_dir = Path(artifacts_dir)
self.raw_dir = self.artifacts_dir / "raw"
create_dir(self.raw_dir)


def load_data(self) -> pd.DataFrame:
"""Load dataset from CSV/parquet. Add more formats if needed."""
try:
if self.data_path.suffix in [".csv"]:
df = pd.read_csv(self.data_path, parse_dates=["date"], infer_datetime_format=True)
elif self.data_path.suffix in [".parquet"]:
df = pd.read_parquet(self.data_path)
else:
raise CustomException(f"Unsupported file format: {self.data_path.suffix}")


logger.info(f"Loaded data with shape: {df.shape}")
return df
except Exception as e:
logger.exception("Failed to load data")
raise


def save_raw(self, df: pd.DataFrame, filename: Optional[str] = "raw_data.csv") -> Path:
out_path = self.raw_dir / filename
df.to_csv(out_path, index=False)
logger.info(f"Saved raw data to: {out_path}")
return out_path