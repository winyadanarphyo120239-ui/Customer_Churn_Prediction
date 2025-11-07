from flask import Flask, render_template, request, redirect, url_for, flash
import joblib, os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = "replace_with_a_secure_key"

# Paths for all models and scaler
MODEL_PATHS = {
    "pkl_model": "model.pkl",
    "random_forest": "best_model_random_forest.joblib",
    "decision_tree": "best_model_decision_tree.joblib"
}
SCALER_PATH = "scaler.pkl"

CATEGORICAL_COLS = ['Gender', 'ContractType', 'InternetService', 'TechSupport']

# Load scaler once
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    else:
        return None

scaler = load_scaler()

def load_model(model_choice):
    path = MODEL_PATHS.get(model_choice)
    if not path or not os.path.exists(path):
        return None
    # Use joblib for .joblib files, pickle for .pkl
    if path.endswith(".pkl"):
        return pickle.load(open(path, 'rb'))
    else:
        return joblib.load(path)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form.get("model", "pkl_model")
    model = load_model(model_choice)

    if model is None:
        flash(f"Model file for '{model_choice}' not found. Please ensure it exists in the project folder.", "danger")
        return redirect(url_for("home"))
    if scaler is None:
        flash("scaler.pkl not found in project folder. Please place your scaler file there.", "danger")
        return redirect(url_for("home"))

    # Collect form inputs
    try:
        age = float(request.form.get("Age", 0))
        gender = request.form.get("Gender", "Male")
        tenure = float(request.form.get("Tenure", 0))
        monthly = float(request.form.get("MonthlyCharges", 0))
        contract = request.form.get("ContractType", "")
        internet = request.form.get("InternetService", "")
        tech = request.form.get("TechSupport", "")
    except ValueError:
        flash("Please enter valid numeric values for Age, Tenure, and MonthlyCharges.", "danger")
        return redirect(url_for("home"))

    row = {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "MonthlyCharges": monthly,
        "ContractType": contract,
        "InternetService": internet,
        "TechSupport": tech
    }
    df = pd.DataFrame([row])

    # Apply same preprocessing as training
    df_processed = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns and df[c].nunique() > 0])

    # Match number of features with model/scaler
    expected_n = None
    if hasattr(model, "n_features_in_"):
        expected_n = int(model.n_features_in_)
    elif hasattr(scaler, "mean_"):
        expected_n = int(len(scaler.mean_))
    else:
        expected_n = df_processed.shape[1]

    if df_processed.shape[1] < expected_n:
        for i in range(expected_n - df_processed.shape[1]):
            df_processed[f"pad_{i}"] = 0.0
    elif df_processed.shape[1] > expected_n:
        df_processed = df_processed.iloc[:, :expected_n]

    X = df_processed.values.astype(float)

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        flash(f"Error while scaling input: {e}", "danger")
        return redirect(url_for("home"))

    # Prediction
    try:
        pred = model.predict(X_scaled)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)[0][1] * 100
            prob = round(prob, 2)
        label = int(pred[0])
        text = "Customer will churn" if label == 1 else "Customer will not churn"
    except Exception as e:
        flash(f"Error during prediction: {e}", "danger")
        return redirect(url_for("home"))

    return render_template("index.html", prediction=text, probability=prob, inputs=row)

# ----------------- ADDITIONAL ROUTES -----------------
@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/insights")
def insights():
    return render_template("insights.html")

@app.route("/about")
def about():
    return render_template("about.html")

# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
