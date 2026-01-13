# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from model_utils import load_pipeline, predict_single, recommend_actions, lime_explain_single
import joblib
import json

app = Flask(__name__)
CORS(app)

SAVED_DIR = "saved_models"
METRICS_PATH = os.path.join(SAVED_DIR, "metrics.json")
PIPELINE_PATH = os.path.join(SAVED_DIR, "pipeline.pkl")
SHAP_SUM_PATH = os.path.join(SAVED_DIR, "shap_summary.json")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error":"no JSON body provided"}), 400
    try:
        result = predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error":"metrics not found. Train the model first."}), 404
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    return jsonify(metrics)

@app.route("/shap-summary", methods=["GET"])
def shap_summary():
    if not os.path.exists(SHAP_SUM_PATH):
        return jsonify({"error":"shap_summary.json not found. Retrain with SHAP enabled."}), 404
    with open(SHAP_SUM_PATH, "r") as f:
        data = json.load(f)
    return jsonify(data)

@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """
    Returns feature importances. Prefer SHAP summary if available, else tries model.feature_importances_
    """
    if os.path.exists(SHAP_SUM_PATH):
        with open(SHAP_SUM_PATH, "r") as f:
            data = json.load(f)
        return jsonify({"source":"shap", "data": data})
    # fallback: try to load pipeline and estimator
    if not os.path.exists(PIPELINE_PATH):
        return jsonify({"error":"pipeline not found. Train model first."}), 404
    pipeline = joblib.load(PIPELINE_PATH)
    try:
        estimator = pipeline.named_steps['clf']
        fi = None
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            # attempt to get feature names from preprocessor (like train.py)
            try:
                pre = pipeline.named_steps['preprocessor']
                num_feats = pre.transformers_[0][2]
                cat_transformer = pre.transformers_[1][1].named_steps['onehot']
                cat_feats = pre.transformers_[1][2]
                cat_names = list(cat_transformer.get_feature_names_out(cat_feats))
                all_feature_names = list(num_feats) + cat_names
            except Exception:
                all_feature_names = [f"f{i}" for i in range(len(importances))]
            top = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
            return jsonify({"source":"feature_importances", "data": [{"feature":k,"importance":float(v)} for k,v in top]})
        else:
            return jsonify({"error":"No feature importances available for this estimator."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no JSON body provided"}), 400
    try:
        result = recommend_actions(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/lime", methods=["POST"])
def lime():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no JSON body provided"}), 400
    try:
        result = lime_explain_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
