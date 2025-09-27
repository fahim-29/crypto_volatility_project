# src/pipeline/prediction_pipeline.py

from pathlib import Path
import pandas as pd
import joblib

from src.utils.logger import get_logger
from src.utils.exception import CustomException

logger = get_logger(__name__)


class PredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model_path = self.artifacts_dir / "models" / "best_pipeline.joblib"

        if not self.model_path.exists():
            raise CustomException(f"Model file not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def predict(self, input_data: pd.DataFrame):
        """Make predictions on new input data."""
        if input_data.empty:
            raise CustomException("Input data for prediction is empty")

        preds = self.pipeline.predict(input_data)
        return preds


if __name__ == "__main__":
    print("🔹 Running prediction pipeline test...")

    # Load some model-ready features (for demo)
    features_path = Path("artifacts/crypto_features_model.csv")
    if not features_path.exists():
        raise CustomException("Model-ready features not found. Run features.py first.")

    df = pd.read_csv(features_path, parse_dates=["date"])

    # Drop target column if present
    target_col = "vol_7d_target_next"
    if target_col in df.columns:
        X_new = df.drop(columns=[target_col, "date"], errors="ignore")
    else:
        X_new = df.drop(columns=["date"], errors="ignore")

    predictor = PredictionPipeline()
    preds = predictor.predict(X_new.head(10))  # test on first 10 rows
    print("✅ Sample predictions:", preds)
