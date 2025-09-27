# Low-Level Design (LLD)

The Low-Level Design explains the internal working of each component in detail.

---

## 1. Components

### a) `src/components/data_ingestion.py`
- **Purpose**: Loads raw cryptocurrency data (CSV/Parquet).
- **Key Functions**:
  - `load_data()` → reads the file, parses dates, returns a DataFrame.
  - `save_raw()` → saves a copy to `artifacts/raw/raw_data.csv`.

---

### b) `src/components/data_transformation.py`
- **Purpose**: Prepares data for model training.
- **Steps**:
  - Detects numerical vs categorical columns.
  - Builds preprocessing pipelines:
    - **Numerical** → SimpleImputer (median), StandardScaler.
    - **Categorical** → SimpleImputer (most frequent), OneHotEncoder.
  - Saves preprocessor to `artifacts/transformer/preprocessor.joblib`.

---

### c) `src/components/model_trainer.py`
- **Purpose**: Train, evaluate, and save models.
- **Key Functions**:
  - `evaluate()` → returns RMSE, MAE, R².
  - `save_model()` → saves best pipeline to `artifacts/models/`.
  - `load_model()` → loads a trained model from disk.

---

## 2. Pipelines

### a) `src/pipeline/training_pipeline.py`
- **Purpose**: Orchestrates end-to-end training.
- **Flow**:
  - Loads dataset (prefers `crypto_features_model.csv` if available).
  - Defines target variable.
  - Splits into train/test sets.
  - Applies preprocessing.
  - Trains RandomForest and XGBoost.
  - Evaluates and selects best model.
  - Saves the trained pipeline.

---

### b) `src/pipeline/prediction_pipeline.py`
- **Purpose**: Runs predictions on new CSV input.
- **Flow**:
  - Loads `best_pipeline.joblib`.
  - Preprocesses input CSV.
  - Generates predictions.
  - Returns results for Flask app.

---

## 3. Utils

### a) `src/utils/utils.py`
- **Purpose**: Shared helper functions.
- **Functions**:
  - `create_dir()` → creates directories.
  - `save_pickle()` & `load_pickle()` → saves/loads serialized objects.
  - `evaluate()` → computes metrics (RMSE, MAE, R²).

---

### b) `src/utils/logger.py`
- **Purpose**: Central logging utility.
- **Function**:
  - `get_logger()` → returns a configured logger with timestamps.

---

### c) `src/utils/exception.py`
- **Purpose**: Custom exception handling.
- **Class**:
  - `CustomException` → improves error messages by wrapping exceptions.

---

## 4. Feature Engineering

### `src/features.py`
- **Purpose**: Creates advanced features for volatility prediction.
- **Steps**:
  - Computes rolling volatility.
  - Computes ATR and other technical indicators.
  - Generates liquidity ratios (volume/marketCap).
- **Outputs**:
  - `artifacts/crypto_features_full.csv`
  - `artifacts/crypto_features_model.csv`

---

## 5. Deployment

### `app.py`
- **Purpose**: Flask web app for prediction.
- **Flow**:
  - User uploads CSV via web form.
  - Backend loads saved model.
  - Runs predictions.
  - Displays results in browser + allows CSV download.
