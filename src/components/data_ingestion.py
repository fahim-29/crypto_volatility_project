from pathlib import Path
import pandas as pd
from typing import Optional

class DataIngestion:
    
    def __init__(self, data_path: str, artifacts_dir: str = "artifacts"):
        self.data_path = Path(data_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.raw_dir = self.artifacts_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        if self.data_path.suffix == ".csv":
            df = pd.read_csv(self.data_path, parse_dates=["date"], infer_datetime_format=True)
        elif self.data_path.suffix == ".parquet":
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        print(f"Loaded data with shape: {df.shape}")
        return df


    def save_raw(self, df: pd.DataFrame, filename: Optional[str] = "raw_data.csv") -> Path:
        out_path = self.raw_dir / filename
        df.to_csv(out_path, index=False)
        print(f"Saved raw data to: {out_path}")
        return out_path

if __name__ == "__main__":
    # Correct path (inside data folder)
    ingestion = DataIngestion(data_path="data/crypto_prices.csv")  
    df = ingestion.load_data()
    print(df.head())
    ingestion.save_raw(df)
