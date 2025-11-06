from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import joblib
import numpy as np
import os
import warnings

# Suppress XGBoost serialization warning
warnings.filterwarnings('ignore', message='.*If you are loading a serialized model.*')

app = FastAPI(title="Financial Health API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# paths
MODEL_PATH = "models/xgb_financial_health.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
CLASS_MAP_PATH = "models/class_mapping.pkl"

# load artifacts
for p in (MODEL_PATH, FEATURES_PATH, CLASS_MAP_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing artifact: {p}")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
class_map = joblib.load(CLASS_MAP_PATH)

# Normalize class_map keys
if all(isinstance(k, str) and k.isdigit() for k in class_map.keys()):
    class_map = {int(k): v for k, v in class_map.items()}

@app.post("/predict")
def predict(payload: Dict[str, Any]):
    # payload is already a dict, no need for .dict()
    payload_dict = payload

    # Normalize input keys
    def norm(s): return "".join(str(s).lower().replace("_", " ").split())
    inv = {norm(k): v for k, v in payload_dict.items()}

    # build ordered row
    row = []
    missing = []
    for feat in features:
        k = norm(feat)
        if k in inv:
            row.append(inv[k])
        else:
            missing.append(feat)
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})

    X = np.array([row])
    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0].tolist()

    # Get classes order
    try:
        classes_order = [int(c) for c in model.classes_]
    except Exception:
        classes_order = list(range(len(probs)))
    labels = [class_map.get(c, str(c)) for c in classes_order]

    # Determine confidence
    try:
        pos = classes_order.index(pred_idx)
        confidence = float(probs[pos])
    except ValueError:
        confidence = float(max(probs))

    # label mapping to canonical frontend labels
    label_map = {
        "Safe": "Healthy",
        "safe": "Healthy",
        "Healthy": "Healthy",
        "healthy": "Healthy",
        "Grey": "At Risk",
        "Gray": "At Risk",
        "grey": "At Risk",
        "gray": "At Risk",
        "At Risk": "At Risk",
        "at risk": "At Risk",
        "atrisk": "At Risk",
        "Distress": "Critical",
        "distress": "Critical",
        "Critical": "Critical",
        "critical": "Critical",
        "Danger": "Critical",
        "danger": "Critical"
    }

    orig_probs = dict(zip(labels, probs))
    mapped_probs = {}
    for k, v in orig_probs.items():
        mapped_key = label_map.get(k, None) or label_map.get(k.lower(), None)
        if mapped_key is None:
            kl = k.lower()
            if "safe" in kl or "health" in kl or "good" in kl:
                mapped_key = "Healthy"
            elif "grey" in kl or "gray" in kl or "risk" in kl:
                mapped_key = "At Risk"
            elif "distress" in kl or "crit" in kl or "danger" in kl:
                mapped_key = "Critical"
            else:
                mapped_key = k
        mapped_probs[mapped_key] = mapped_probs.get(mapped_key, 0.0) + float(v)

    total = sum(mapped_probs.values()) or 1.0
    for mk in mapped_probs:
        mapped_probs[mk] = mapped_probs[mk] / total

    raw_status = class_map.get(pred_idx, str(pred_idx))
    mapped_status = label_map.get(raw_status, None) or label_map.get(raw_status.lower(), None) or raw_status
    if mapped_status not in ("Healthy", "At Risk", "Critical"):
        s = str(mapped_status).lower()
        if "safe" in s or "health" in s or "good" in s:
            mapped_status = "Healthy"
        elif "grey" in s or "gray" in s or "risk" in s:
            mapped_status = "At Risk"
        elif "distress" in s or "crit" in s or "danger" in s:
            mapped_status = "Critical"

    return {
        "status": mapped_status,
        "confidence": confidence,
        "probs": mapped_probs
    }