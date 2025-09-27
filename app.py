# app.py
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils.exception import CustomException

# Config
UPLOAD_FOLDER = "uploads"
PREDICTIONS_FOLDER = "artifacts/predictions"
ALLOWED_EXTENSIONS = {"csv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-key"  # set a secure key for production
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # check file upload
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(saved_path)

        # Read CSV
        try:
            df = pd.read_csv(saved_path, parse_dates=["date"], infer_datetime_format=True)
        except Exception:
            # try without parse_dates if date column isn't present
            df = pd.read_csv(saved_path)

        # Drop target/date columns if present
        target_col = "vol_7d_target_next"
        if target_col in df.columns:
            X_new = df.drop(columns=[target_col, "date"], errors="ignore")
        else:
            X_new = df.drop(columns=["date"], errors="ignore")

        # Limit number of rows to predict (optional)
        try:
            max_rows = int(request.form.get("max_rows", 100))
        except Exception:
            max_rows = 100
        X_sample = X_new.head(max_rows)

        # Run prediction with PredictionPipeline
        try:
            predictor = PredictionPipeline()
            preds = predictor.predict(X_sample)
        except CustomException as ce:
            flash(str(ce))
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Prediction failed: {e}")
            return redirect(url_for("index"))

        # Append predictions and save CSV
        out_df = X_sample.copy()
        out_df["prediction"] = preds
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_filename = f"predictions_{timestamp}.csv"
        out_path = Path(PREDICTIONS_FOLDER) / out_filename
        out_df.to_csv(out_path, index=False)

        # Render results page with sample predictions and download link
        sample_results = out_df.head(20).to_dict(orient="records")
        return render_template("result.html", results=sample_results, download_file=out_filename)

    else:
        flash("Allowed file types: csv")
        return redirect(url_for("index"))

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(PREDICTIONS_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
