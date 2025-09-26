# Cryptocurrency Volatility Prediction

This project applies machine learning to predict cryptocurrency market volatility.  
We use historical price data to model and forecast volatility, helping in risk management and trading strategies.

---

## 📂 Project Structure
- `src/` → source code
- `data/` → raw and processed datasets
- `notebooks/` → Jupyter notebooks for EDA and experiments
- `artifacts/` → saved models, scalers, processed data
- `logs/` → logging for debugging
- `tests/` → unit tests
- `main.py` → entry point for the pipeline

---

## 📊 Dataset
- **Source:** Cryptocurrency historical OHLC, volume, and market cap data  
- **Features used:**  
  - `open`, `high`, `low`, `close`  
  - `volume`, `marketCap`  
  - Engineered features:  
    - Log returns  
    - Rolling volatility (7d, 30d)  
    - Moving averages (7d, 30d)  
    - ATR (Average True Range)  
    - Liquidity ratios  

- **Target variable:**  
  - `vol_7d_target_next` → realized volatility over the next 7 days  

---

## 🔬 Methodology
1. **Data Preprocessing**  
   - Handled missing values, cleaned anomalies  
   - Normalized/scaled features  
   - Removed duplicates  

2. **Feature Engineering**  
   - Rolling volatility (7-day, 30-day)  
   - Moving averages  
   - Liquidity ratios (`volume/marketCap`)  
   - Technical indicators (ATR, TR, etc.)  

3. **Modeling**  
   - Linear Regression (baseline)  
   - Random Forest (nonlinear baseline)  
   - XGBoost (gradient boosting)  
   - Random Forest (hyperparameter tuned)  

4. **Evaluation Metrics**  
   - RMSE (Root Mean Squared Error)  
   - MAE (Mean Absolute Error)  
   - R² Score  

---

## 📈 Results

| Model                  | RMSE   | MAE    | R²    |
|-------------------------|--------|--------|-------|
| Linear Regression       | 0.0305 | 0.0181 | -0.31 |
| Random Forest (default) | 0.0067 | 0.0028 | 0.94  |
| XGBoost (default)       | 0.0077 | 0.0030 | 0.92  |
| Random Forest (tuned)   | 0.0066 | 0.0029 | 0.94  |

- ✅ **Random Forest (tuned)** performed best with an R² ≈ **93.7%**  

---

## 📊 Model Validation
- Actual vs Predicted volatility plots showed a **close match** over the test set.  
- Residuals are centered around zero with few outliers → model is not biased.  
- Outliers reflect market shocks (expected in crypto).  

---

## 🚀 Next Steps
- Further hyperparameter tuning (e.g., XGBoost, LightGBM)  
- Try ensemble methods (stacking models)  
- Deploy locally with **Streamlit** or **Flask**  
- Host on **Streamlit Cloud** or **Heroku** for public access  

---

## ⚙️ How to Run

```bash
# Clone repository
git clone https://github.com/fahim-29/crypto_volatility_project.git

# Navigate into project folder
cd crypto_volatility_project

# Install dependencies
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/
