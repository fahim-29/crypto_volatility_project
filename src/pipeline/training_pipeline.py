"""Orchestrator script to run ingestion -> transformation -> training end-to-end.
Run: `python -m src.pipeline.training_pipeline` from project root or add entrypoint.
"""
from pathlib import Path
import argparse
import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import get_logger
from sklearn.model_selection import train_test_split


logger = get_logger(__name__)




def run_training(data_path: str, artifacts_dir: str = "artifacts"):
# 1) ingest
ingestion = DataIngestion(data_path=data_path, artifacts_dir=artifacts_dir)
df = ingestion.load_data()
ingestion.save_raw(df)


# 2) prepare train/test split (time-based split recommended in notebook)
# NOTE: for simplicity below we use sklearn train_test_split; replace with time-based split if needed.
target_col = "vol_7d_target_next"
X = df.drop(columns=[target_col, "date"], errors="ignore")
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# 3) build / fit preprocessor
numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X_train.columns if c not in numerical_cols]


transformer = DataTransformation(artifacts_dir=artifacts_dir)
preprocessor = transformer.build_preprocessor(numerical_cols, categorical_cols)
preprocessor.fit(X_train)
transformer.save_preprocessor(preprocessor)


# 4) define pipelines for models (example: RF and XGB)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


rf_pipeline = Pipeline(steps=[
("preprocessor", preprocessor),
("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
])


xgb_pipeline = Pipeline(steps=[
("preprocessor", preprocessor),
("model", XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1))
])


# 5) train & evaluate
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)


trainer = ModelTrainer(artifacts_dir=artifacts_dir)
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_xgb = xgb_pipeline.predict(X_test)


rf_metrics = trainer.evaluate(y_test, y_pred_rf)
xgb_metrics = trainer.evaluate(y_test, y_pred_xgb)


logger.info(f"RF metrics: {rf_metrics}")
logger.info(f"XGB metrics: {xgb_metrics}")


# 6) persist best model (choose by r2)
best_pipeline = rf_pipeline if rf_metrics["r2"] >= xgb_metrics["r2"] else xgb_pipeline
trainer.save_model(best_pipeline, filename="best_pipeline.joblib")




if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to dataset CSV or parquet")
parser.add_argument("--artifacts_dir", default="artifacts", help="Where to save models and transformers")
args = parser.parse_args()


run_training(data_path=args.data, artifacts_dir=args.artifacts_dir)