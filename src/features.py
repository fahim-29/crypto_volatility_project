# src/features.py
"""
Feature engineering script for Crypto Volatility project.

Usage:
    conda activate crypto_volatility_env
    python src/features.py

This script expects: data/crypto_prices.csv
It produces: artifacts/crypto_features_full.csv and artifacts/crypto_features_model.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    if path is None:
        path = PROJECT_ROOT / "data" / "crypto_prices.csv"
    return pd.read_csv(path)


def ensure_datetime(crypto_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure date/time columns are datetime dtype.
    This is tolerant: if they are already datetime, this does nothing.
    """
    if 'timestamp' in crypto_prices.columns and not pd.api.types.is_datetime64_any_dtype(crypto_prices['timestamp']):
        crypto_prices['timestamp'] = pd.to_datetime(crypto_prices['timestamp'], utc=True, errors='coerce')
    if 'date' in crypto_prices.columns and not pd.api.types.is_datetime64_any_dtype(crypto_prices['date']):
        crypto_prices['date'] = pd.to_datetime(crypto_prices['date'], errors='coerce')
    return crypto_prices


def feature_engineer(crypto_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add features useful for volatility prediction.

    Uses the variable name crypto_prices throughout (as you requested).
    Assumes columns: ['open','high','low','close','volume','marketCap','timestamp','crypto_name','date']
    """
    # 1) Sort by crypto and date to make rolling ops correct within each crypto
    crypto_prices = crypto_prices.sort_values(['crypto_name', 'date']).reset_index(drop=True)

    # 2) Log price and log return (stabilize scale)
    crypto_prices['log_price'] = np.log1p(crypto_prices['close'])
    crypto_prices['log_return'] = crypto_prices.groupby('crypto_name')['log_price'].diff()

    # 3) Rolling volatility & moving averages per crypto (7-day and 30-day)
    crypto_prices['vol_7d'] = crypto_prices.groupby('crypto_name')['log_return'] \
                                           .rolling(window=7, min_periods=1).std() \
                                           .reset_index(level=0, drop=True)
    crypto_prices['vol_30d'] = crypto_prices.groupby('crypto_name')['log_return'] \
                                            .rolling(window=30, min_periods=1).std() \
                                            .reset_index(level=0, drop=True)

    crypto_prices['ma_7'] = crypto_prices.groupby('crypto_name')['close'] \
                                         .rolling(window=7, min_periods=1).mean() \
                                         .reset_index(level=0, drop=True)
    crypto_prices['ma_30'] = crypto_prices.groupby('crypto_name')['close'] \
                                          .rolling(window=30, min_periods=1).mean() \
                                          .reset_index(level=0, drop=True)

    # 4) Liquidity ratio: volume / marketCap (avoid division by zero)
    crypto_prices['liquidity'] = crypto_prices['volume'] / (crypto_prices['marketCap'].replace(0, np.nan) + 1e-9)

    # 5) True Range (TR) and ATR(14) computed per crypto safely
    def _compute_tr_atr(group: pd.DataFrame) -> pd.DataFrame:
        hl = group['high'] - group['low']
        # previous close within the same crypto
        prev_close = group['close'].shift(1)
        hc = (group['high'] - prev_close).abs()
        lc = (group['low'] - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        group['tr'] = tr
        group['atr_14'] = tr.rolling(window=14, min_periods=1).mean()
        return group

    crypto_prices = crypto_prices.groupby('crypto_name', group_keys=False).apply(_compute_tr_atr)

    # 6) Create forward (next-day) target: next day vol_7d (you can change target as needed)
    crypto_prices['vol_7d_target_next'] = crypto_prices.groupby('crypto_name')['vol_7d'].shift(-1)

    # 7) (Optional) drop any helper columns you don't want saved, e.g. keep 'tr' or drop it
    # crypto_prices = crypto_prices.drop(columns=['tr'])

    return crypto_prices


def save_processed(crypto_prices: pd.DataFrame):
    """Save both a full feature file and a model-ready file (rows with NaNs dropped)."""
    out_dir = PROJECT_ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "crypto_features_full.csv"
    crypto_prices.to_csv(full_path, index=False)
    print(f"[+] Saved full feature table to: {full_path}")

    # For model training we typically drop rows where engineered features or the target are NaN
    required_cols = ['log_return', 'vol_7d', 'ma_7', 'ma_30', 'liquidity', 'atr_14', 'vol_7d_target_next']
    model_df = crypto_prices.dropna(subset=required_cols)
    model_path = out_dir / "crypto_features_model.csv"
    model_df.to_csv(model_path, index=False)
    print(f"[+] Saved model-ready table to: {model_path} (rows with NA in required cols dropped)")
    print(f"[+] Rows retained for modeling: {len(model_df)} / {len(crypto_prices)}")


def main():
    print("Loading raw data...")
    crypto_prices = load_data()
    print("Checking/ensuring datetime columns (no rework if already done)...")
    crypto_prices = ensure_datetime(crypto_prices)

    print("Running feature engineering...")
    crypto_prices = feature_engineer(crypto_prices)

    print("Saving processed outputs...")
    save_processed(crypto_prices)
    print("Done.")


if __name__ == "__main__":
    main()
