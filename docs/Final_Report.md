# Final Report: Cryptocurrency Volatility Prediction

## 1. Problem Statement
Cryptocurrency markets are highly volatile.  
The goal of this project was to build a machine learning model that predicts cryptocurrency volatility using historical price, volume, and market cap data.  
This helps traders and institutions manage risk, improve portfolio allocation, and anticipate market instability.

---

## 2. Dataset Information
- Source: Historical cryptocurrency prices dataset (50+ cryptocurrencies).  
- Features included:
  - Date, Symbol, Open, High, Low, Close
  - Trading Volume
  - Market Capitalization
- Rows: ~72,000  
- Processed down to ~70,000 rows after cleaning.

---

## 3. Approach
1. **Data Ingestion**  
   - Loaded raw CSV.  
   - Saved a clean copy to `artifacts/raw/`.  

2. **Data Transformation**  
   - Built preprocessing pipeline:
     - Numerical: imputation (median), scaling.  
     - Categorical: imputation (most frequent), one-hot encoding.  

3. **Feature Engineering**  
   - Computed rolling volatility, ATR, liquidity ratios.  
   - Produced model-ready dataset `crypto_features_model.csv`.  

4. **Model Training**  
   - Models: Linear Regression (baseline), RandomForest, XGBoost.  
   - Evaluation metrics: RMSE, MAE, R².  
   - RandomForest performed best (R² ≈ 0.95).  
   - Best pipeline saved to `artifacts/models/best_pipeline.joblib`.  

5. **Prediction Pipeline**  
   - Loads best model pipeline.  
   - Generates predictions on new input CSV.  

6. **Deployment**  
   - Flask web app built.  
   - User uploads CSV file.  
   - Predictions returned in browser + downloadable CSV.  

---

## 4. Results
- **RandomForest** outperformed XGBoost and baseline models.  
- **Best model performance**:  
  - RMSE: ~0.0057  
  - MAE: ~0.0022  
  - R²: ~0.95  

---

## 5. Conclusion
- Successfully built a full ML pipeline for cryptocurrency volatility prediction.  
- Modular design ensures:
  - Reusability (components can be extended easily).  
  - Scalability (new models/features can be added).  
- Flask deployment demonstrates practical usability.  

---

## 6. Future Enhancements
- Integrate LSTM or deep learning models for time-series volatility prediction.  
- Extend Flask app to allow interactive visualizations.  
- Deploy to cloud (AWS/GCP/Heroku) for global accessibility.  
