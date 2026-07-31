"""
app.py
Task 3: API Development

Flask REST API that loads the trained model and serves predictions.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model artifact (model + scaler + feature order)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]


@app.route("/", methods=["GET"])
def home():
    # Simple HTML form (optional, templates/index.html)
    return render_template("index.html", features=feature_names)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts patient details as JSON, e.g.:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    Returns:
    { "prediction": "Heart Disease Detected" }
    """
    try:
        data = request.get_json(force=True)

        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Preserve exact training feature order
        import pandas as pd
        input_values = [float(data[f]) for f in feature_names]
        input_df = pd.DataFrame([input_values], columns=feature_names)
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        result = "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected"

        return jsonify({
            "prediction": result,
            "probability": round(float(proba), 4)
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
