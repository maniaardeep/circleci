# src/predict.py
from flask import Flask, jsonify
import pickle

app = Flask(__name__)
MODEL_PATH = "models/model.pkl"

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict")
def predict():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        # return a dummy prediction
        return jsonify({"pred": int(model.predict([[5.1,3.5,1.4,0.2]])[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
