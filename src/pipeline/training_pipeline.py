# src/pipeline/training_pipeline.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def run_training(data_path="data/crypto_prices.csv", artifacts_dir="artifacts"):
    print("🔹 Training pipeline started...")

    artifacts_dir = Path(artifacts_dir)

    # 1) Prefer model-ready features if available
    model_features_path = artifacts_dir / "crypto_features_model.csv"
    if model_features_path.exists():
        print(f"📂 Using pre-computed features: {model_features_path}")
        df = pd.read_csv(model_features_path, parse_dates=["date"])
    else:
        # Fall back to ingestion + raw data
        ingestion = DataIngestion(data_path=data_path, artifacts_dir=artifacts_dir)
        df = ingestion.load_data()
        ingestion.save_raw(df)
        print("✅ Ingestion finished! Shape:", df.shape)

    # 2) Define target
    target_col = "vol_7d_target_next"
    if target_col not in df.columns:
        print(f"⚠️ Target '{target_col}' not found. Using 'close' as fallback.")
        target_col = "close"

    X = df.drop(columns=[target_col, "date"], errors="ignore")
    y = df[target_col]

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 4) Transformation
    transformer = DataTransformation(artifacts_dir=artifacts_dir)
    numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numerical_cols]

    preprocessor = transformer.build_preprocessor(numerical_cols, categorical_cols)
    preprocessor.fit(X_train)
    transformer.save_preprocessor(preprocessor)
    print("✅ Transformation finished! Numerical:", len(numerical_cols), "Categorical:", len(categorical_cols))

    # 5) Define candidate models
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])

    xgb_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 6) Train both models
    print("🔹 Training RandomForest...")
    rf_pipeline.fit(X_train, y_train)
    print("🔹 Training XGBoost...")
    xgb_pipeline.fit(X_train, y_train)

    # 7) Evaluate
    trainer = ModelTrainer(artifacts_dir=artifacts_dir)
    rf_metrics = trainer.evaluate(y_test, rf_pipeline.predict(X_test))
    xgb_metrics = trainer.evaluate(y_test, xgb_pipeline.predict(X_test))

    print("RandomForest metrics:", rf_metrics)
    print("XGBoost metrics:", xgb_metrics)

    # 8) Save best model
    if xgb_metrics["r2"] >= rf_metrics["r2"]:
        best_pipeline = xgb_pipeline
        print("✅ XGBoost selected as best model")
    else:
        best_pipeline = rf_pipeline
        print("✅ RandomForest selected as best model")

    trainer.save_model(best_pipeline, filename="best_pipeline.joblib")

    print("✅ Training pipeline finished.")


if __name__ == "__main__":
    run_training()
