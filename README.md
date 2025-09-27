# 🚀 Cryptocurrency Volatility Prediction

## 📌 Project Overview
Cryptocurrency markets are notoriously volatile, making it crucial for traders and institutions to forecast volatility for **risk management, portfolio allocation, and informed decision-making**.  

This project builds a **complete ML pipeline** to predict cryptocurrency volatility based on historical OHLC (Open, High, Low, Close) prices, trading volume, and market capitalization.  

The system covers:  
✅ Data ingestion → transformation → feature engineering  
✅ Model training (RandomForest, XGBoost) → evaluation  
✅ Saving best pipeline → prediction  
✅ Local deployment with a Flask web app  

---

## 📂 Folder Structure


crypto_volatility_project/
│── data/                  # Dataset(s)
│   └── crypto_prices.csv
│
│── docs/                  # Documentation (HLD, LLD, Final Report)
│   ├── HLD.md
│   ├── LLD.md
│   └── Final_Report.md
│
│── flowchart/             # System flowcharts (DOT + PNG)
│   ├── data_ingestion.dot / .png
│   ├── data_transformation.dot / .png
│   ├── model_trainer.dot / .png
│   ├── pipeline_flow.dot / .png
│   └── prediction_pipeline.dot / .png
│
│── notebooks/             # Jupyter notebooks for EDA & model training
│   ├── 01_EDA_CryptoVolatility.ipynb
│   └── Model_Training.ipynb
│
│── src/                   # Source code
│   ├── components/        # Ingestion, transformation, trainer
│   ├── pipeline/          # Training & prediction pipelines
│   ├── utils/             # Logger, exception handler, helpers
│   └── features.py        # Feature engineering
│
│── static/                # CSS / static assets for Flask
│── templates/             # HTML templates for Flask
│── app.py                 # Flask web app
│── main.py                # Entry point (optional)
│── requirements.txt       # Project dependencies
│── README.md              # Project documentation (this file)



## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/fahim-29/crypto_volatility_project.git
cd crypto_volatility_project

# 2. Create and activate a virtual environment
conda create -n crypto_volatility_env python=3.11 -y
conda activate crypto_volatility_env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the training pipeline (train models and save best one)
python -m src.pipeline.training_pipeline

# 5. (Optional) Test prediction pipeline
python -m src.pipeline.prediction_pipeline

# 6. Start the Flask web app
python app.py



## 📊 Results
- **Best Model**: RandomForest Regressor  
- **Performance**:  
  - RMSE: ~0.0057  
  - MAE: ~0.0022  
  - R²: ~0.95  

---

## 📖 Documentation
- [High-Level Design (HLD)](docs/HLD.md)  
- [Low-Level Design (LLD)](docs/LLD.md)  
- [Final Report](docs/Final_Report.md)  

---

## 📌 Flowcharts
### System Architecture
![Pipeline Flow](flowchart/pipeline_flow.png)

### Data Ingestion
![Data Ingestion Flow](flowchart/data_ingestion.png)

### Data Transformation
![Data Transformation Flow](flowchart/data_transformation.png)

### Model Training
![Model Trainer Flow](flowchart/model_trainer.png)

### Prediction Pipeline
![Prediction Pipeline Flow](flowchart/prediction_pipeline.png)

---

## 🚀 Future Improvements
- 🔹 Integrate LSTM/GRU for advanced time-series volatility forecasting.  
- 🔹 Deploy Flask app to **Heroku / AWS / GCP** for global access.  
- 🔹 Add dashboards with **Plotly/Dash or Streamlit** for better visualization.  

---

## 🙌 Acknowledgements
- Dataset source: Public cryptocurrency market data  
- Libraries: `pandas`, `scikit-learn`, `xgboost`, `flask`, `joblib`  
- Mentor guidance and project inspiration from real-world ML workflows  
