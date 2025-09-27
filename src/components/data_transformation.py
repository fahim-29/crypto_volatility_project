from pathlib import Path
import pandas as pd
import joblib
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class DataTransformation:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.transformer_dir = self.artifacts_dir / "transformer"
        self.transformer_dir.mkdir(parents=True, exist_ok=True)

    def build_preprocessor(self, numerical_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        # Numerical pipeline
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ])

        # Combine
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        print("✅ Preprocessor built successfully.")
        return preprocessor

    def save_preprocessor(self, preprocessor, filename: str = "preprocessor.joblib") -> Path:
        out_path = self.transformer_dir / filename
        joblib.dump(preprocessor, out_path)
        print(f"✅ Saved preprocessor to: {out_path}")
        return out_path


if __name__ == "__main__":
    # --- Test block for DataTransformation ---
    df = pd.read_csv("artifacts/raw/raw_data.csv")

    # If target column exists, drop it
    target_col = "vol_7d_target_next"
    if target_col in df.columns:
        X = df.drop(columns=[target_col, "date"], errors="ignore")
    else:
        print(f"⚠️ Target column '{target_col}' not found. Using all columns except 'date'.")
        X = df.drop(columns=["date"], errors="ignore")

    # Separate numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numerical_cols]

    print("Numerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)

    # Build preprocessor
    transformer = DataTransformation()
    preprocessor = transformer.build_preprocessor(numerical_cols, categorical_cols)

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    print("✅ Transformation complete. Transformed shape:", X_transformed.shape)

    # Save preprocessor
    transformer.save_preprocessor(preprocessor)
