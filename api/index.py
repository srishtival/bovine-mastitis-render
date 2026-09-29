import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import pickle
import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import CowSensorInput, CowRecordInput, PredictionResponse, HerdRiskInput, HerdRiskResponse
from src.api.db import (
    insert_observation_with_prediction,
    fetch_latest_record_for_cow,
    fetch_latest_record_per_cow,
    record_to_cow_sensor_input_dict,
    get_connection,
    TABLE,
)
from src.explainability.alert_generator import generate_clinician_alert
from src.herd.herd_engine import calculate_herd_risk_summary

app = FastAPI(
    title="Bovine Mastitis Early Forecasting API",
    description="Integrated ML API for Individual Cow Mastitis Risk Prediction, Herd-Level Analytics, and Clinician Alerts.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
MODEL_DIR = ROOT_DIR / "models"

# Load trained model artifacts
MODEL_DIR = ROOT_DIR / "models"
FEATURE_PATH = MODEL_DIR / "mastitis_features_final.pkl"
MODEL_PATHS = {
    7: {
        "score": MODEL_DIR / "mastitis_risk_score_7d_final.pkl",
        "label": MODEL_DIR / "mastitis_risk_label_7d_final.pkl",
    },
    14: {
        "score": MODEL_DIR / "mastitis_risk_score_14d_final.pkl",
        "label": MODEL_DIR / "mastitis_risk_label_14d_final.pkl",
    },
}
MODEL_FEATURES = [
    'milk_ec', 'milk_temperature_c', 'udder_temperature_c', 'activity_index',
    'milk_colour_code',
    'milk_ec_baseline', 'milk_temp_baseline_c', 'udder_temp_baseline_c',
    'activity_baseline', 'milk_ec_deviation', 'milk_temp_deviation_c',
    'udder_temp_deviation_c', 'activity_deviation', 'milk_ec_slope',
    'udder_temp_slope', 'activity_slope', 'milk_ec_variability',
    'udder_temp_variability', 'activity_variability',
]

@lru_cache(maxsize=1)
def get_model_features() -> list[str]:
    with FEATURE_PATH.open("rb") as handle:
        features = pickle.load(handle)
    if features != MODEL_FEATURES:
        raise RuntimeError(f"Model feature artifact does not match API contract: {features}")
    return features


@lru_cache(maxsize=2)
def get_loaded_models(forecast_horizon_days: int) -> dict[str, Any]:
    if forecast_horizon_days not in MODEL_PATHS:
        raise ValueError("forecast_horizon_days must be 7 or 14")

    get_model_features()
    paths = MODEL_PATHS[forecast_horizon_days]
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")

    try:
        return {
            name: pickle.load(path.open("rb"))
            for name, path in paths.items()
        }
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the XGBoost artifacts. Install the XGBoost version used "
            "to train the models (the requirements pin the compatible major version)."
        ) from exc

@app.get("/")
def root():
    return {
        "status": "active",
        "system": "Bovine Mastitis Predictive Modelling API",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    models = {}
    load_error = None
    for horizon in MODEL_PATHS:
        try:
            get_loaded_models(horizon)
            models[str(horizon)] = True
        except Exception as exc:
            models[str(horizon)] = False
            load_error = str(exc)

    db_ok = False
    db_error = None
    try:
        get_connection().execute("SELECT 1")
        db_ok = True
    except Exception as exc:
        db_error = str(exc)

    return {
        "status": "healthy" if (all(models.values()) and db_ok) else "degraded",
        "model_loaded": all(models.values()),
        "models_loaded": models,
        "database_connected": db_ok,
        "database_error": db_error,
        "model_features": MODEL_FEATURES,
        "load_error": load_error,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.post("/predict_cow", response_model=PredictionResponse)
def predict_individual_cow(inp: CowSensorInput):
    horizon = inp.forecast_horizon_days
    try:
        models = get_loaded_models(horizon)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    feature_values = build_model_features(inp)
    features = pd.DataFrame([feature_values], columns=get_model_features())
    # The XGBRegressor score model outputs a 0-100 risk index (training data:
    # LOW < 35, WATCH/Moderate 35-65, HIGH >= 65); normalise to [0, 1].
    score = float(np.asarray(models["score"].predict(features)).reshape(-1)[0])
    prob = float(min(1.0, max(0.0, score / 100.0)))
    predicted_label = int(np.asarray(models["label"].predict(features)).reshape(-1)[0])

    top_factors = []
    if feature_values['milk_ec_deviation'] > 0:
        top_factors.append(f"Milk conductivity above baseline ({feature_values['milk_ec_deviation']:.2f})")
    if feature_values['udder_temp_deviation_c'] > 0:
        top_factors.append(f"Udder temperature above baseline ({feature_values['udder_temp_deviation_c']:.2f} C)")
    if feature_values['activity_deviation'] < 0:
        top_factors.append(f"Activity below baseline ({abs(feature_values['activity_deviation']):.2f})")
    clinical_override = inp.milk_colour_code >= 3
    if clinical_override:
        top_factors.insert(0, "Observed blood-tinged or clotted milk triggers a high-risk clinical alert; the model forecast tier is reported separately.")
    if not top_factors:
        top_factors = ["All sensor values within normal baseline ranges."]

    alert = generate_clinician_alert(
        cow_id=inp.cow_id,
        farmer_id=inp.farmer_id,
        probability=prob,
        uncertainty=0.05,
        forecast_horizon=horizon,
        top_factors=top_factors,
        clinical_override=clinical_override,
    )
    
    return alert


def build_model_features(inp: CowSensorInput) -> dict[str, float]:
    """Build the model's derived features from the client inputs."""
    feature_values = {
        'milk_ec': inp.milk_ec,
        'milk_temperature_c': inp.milk_temperature_c,
        'udder_temperature_c': inp.udder_temperature_c,
        'activity_index': inp.activity_index,
        'milk_colour_code': float(inp.milk_colour_code),
        'milk_ec_baseline': inp.milk_ec_baseline,
        'milk_temp_baseline_c': inp.milk_temp_baseline_c,
        'udder_temp_baseline_c': inp.udder_temp_baseline_c,
        'activity_baseline': inp.activity_baseline,
        'milk_ec_deviation': inp.milk_ec - inp.milk_ec_baseline,
        'milk_temp_deviation_c': inp.milk_temperature_c - inp.milk_temp_baseline_c,
        'udder_temp_deviation_c': inp.udder_temperature_c - inp.udder_temp_baseline_c,
        'activity_deviation': inp.activity_index - inp.activity_baseline,
        'milk_ec_slope': 0.0,
        'udder_temp_slope': 0.0,
        'activity_slope': 0.0,
        'milk_ec_variability': 0.0,
        'udder_temp_variability': 0.0,
        'activity_variability': 0.0,
    }
    return feature_values

@app.post("/append_cow_record")
def append_cow_record(inp: CowRecordInput):
    prediction = predict_individual_cow(inp)

    try:
        row_id = insert_observation_with_prediction(inp, prediction)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to store observation in the database: {exc}",
        ) from exc

    return {
        'status': 'saved',
        'storage': 'postgresql',
        'table': TABLE,
        'record_id': row_id,
        'cow_id': inp.cow_id,
        **prediction,
    }


def get_latest_record_for_cow(cow_id: str) -> CowRecordInput:
    try:
        record = fetch_latest_record_for_cow(cow_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    if record is None:
        raise HTTPException(status_code=404, detail=f"No saved records found for cow {cow_id}")

    return CowRecordInput(**record_to_cow_sensor_input_dict(record))


def get_all_latest_records_for_cows() -> list[CowRecordInput]:
    try:
        records = fetch_latest_record_per_cow()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

    return [
        CowRecordInput(**record_to_cow_sensor_input_dict(record))
        for record in records
    ]


@app.get('/cow_latest_record/{cow_id}', response_model=CowRecordInput)
def get_recent_cow_record(cow_id: str):
    return get_latest_record_for_cow(cow_id)


@app.get('/cow_records/latest', response_model=list[CowRecordInput])
def get_latest_cow_records():
    return get_all_latest_records_for_cows()


def build_herd_risk_summary(farm_id: str, cows: list[CowSensorInput]):
    predictions = []
    for cow in cows:
        res = predict_individual_cow(cow)
        predictions.append({
            'cow_id': cow.cow_id,
            'mastitis_risk_probability': res['mastitis_risk_probability'],
            'risk_category': res['risk_category']
        })

    if not predictions:
        raise HTTPException(status_code=404, detail='No saved records found in the database.')

    df_preds = pd.DataFrame(predictions)
    summary = calculate_herd_risk_summary(df_preds)

    anomaly_alert = summary['high_risk_pct'] > 15.0

    return {
        "farm_id": farm_id,
        "total_cows_monitored": summary['total_cows_monitored'],
        "high_risk_count": summary['high_risk_count'],
        "high_risk_pct": summary['high_risk_pct'],
        "moderate_risk_count": summary['moderate_risk_count'],
        "low_risk_count": summary['low_risk_count'],
        "no_risk_count": summary['no_risk_count'],
        "herd_health_index": summary['herd_health_index'],
        "anomaly_alert": anomaly_alert
    }


@app.post("/herd_risk", response_model=HerdRiskResponse)
def evaluate_herd_risk(inp: HerdRiskInput):
    return build_herd_risk_summary(inp.farm_id, inp.cows)


@app.get('/herd_risk_db', response_model=HerdRiskResponse)
def evaluate_herd_risk_from_db():
    records = get_all_latest_records_for_cows()
    if not records:
        return {
            "farm_id": "FARM_001",
            "total_cows_monitored": 0,
            "high_risk_count": 0,
            "high_risk_pct": 0.0,
            "moderate_risk_count": 0,
            "low_risk_count": 0,
            "no_risk_count": 0,
            "herd_health_index": 100.0,
            "anomaly_alert": False,
        }

    cows = list(records)  # already CowRecordInput (subclass of CowSensorInput)
    return build_herd_risk_summary('FARM_001', cows)
