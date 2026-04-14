"""
ml/predict.py — Inferencia en Batch y en Tiempo Real
=====================================================
Carga el modelo serializado más reciente y expone una función de predicción
usable tanto por la API (tiempo real) como por jobs batch (producción).
"""

import glob
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

MODELS_DIR = CFG["paths"]["models"]
MANIFEST   = CFG["paths"]["manifest"]


def load_latest_model() -> dict:
    """
    Carga el modelo más reciente desde el manifest.
    Si el manifest no existe, busca el archivo .pkl más reciente en MODELS_DIR.
    """
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
        if "model" in manifest:
            model_path = manifest["model"]["path"]
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    return pickle.load(f)

    # Fallback: buscar el .pkl más reciente
    pkl_files = sorted(glob.glob(f"{MODELS_DIR}/*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No se encontró modelo en {MODELS_DIR}. Ejecuta ml/train.py primero."
        )
    with open(pkl_files[-1], "rb") as f:
        logger.warning(f"Cargando modelo desde fallback: {pkl_files[-1]}")
        return pickle.load(f)


def predict(
    features: pd.DataFrame,
    model_artifact: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Genera predicciones de churn para un DataFrame de features.

    Args:
        features: DataFrame con las mismas columnas que la feature matrix de entrenamiento.
        model_artifact: Dict con pipeline y threshold (si es None, carga el más reciente).

    Returns:
        DataFrame con columnas: customer_id, churn_probability, churn_flag, model_version
    """
    if model_artifact is None:
        model_artifact = load_latest_model()

    pipeline   = model_artifact["pipeline"]
    threshold  = model_artifact["optimal_threshold"]
    version    = model_artifact["model_version"]

    # Validar que las features requeridas estén presentes
    required = model_artifact["num_features"] + model_artifact["cat_features"]
    missing  = [c for c in required if c not in features.columns]
    if missing:
        raise ValueError(f"Faltan features requeridas: {missing}")

    # Inferencia
    proba = pipeline.predict_proba(features)[:, 1]
    flag  = (proba >= threshold).astype(int)

    result = pd.DataFrame({
        "customer_id":       features.index,
        "churn_probability": np.round(proba, 4),
        "churn_flag":        flag,
        "model_version":     version,
        "decision_threshold": threshold,
    })
    return result


if __name__ == "__main__":
    # Demo: carga y predice sobre datos del feature store
    import sqlite3
    db_path = CFG["paths"]["db"]
    conn = sqlite3.connect(db_path)
    sample = pd.read_sql_query(
        "SELECT * FROM ml_feature_store LIMIT 10", conn
    ).set_index("customer_id")
    conn.close()

    if not sample.empty:
        preds = predict(sample)
        print(preds)
    else:
        logger.warning("Feature store vacío. Ejecuta el ETL primero.")
