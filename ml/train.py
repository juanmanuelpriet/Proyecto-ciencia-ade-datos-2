"""
ml/train.py — Entrenamiento del Modelo de Churn
================================================
OBJETIVO PRIMARIO: Maximizar RECALL en la clase positiva (churn=1).
Un falso negativo (perdemos un churner) cuesta más que un falso positivo
(enviamos cupón a alguien que no se iba).

ESTRATEGIA anti-desbalanceo (multi-capa):
1. SMOTE en el train set → genera muestras sintéticas de la clase minoritaria
2. scale_pos_weight en XGBoost → penaliza errores en clase positiva
3. Ajuste de umbral de decisión post-entrenamiento → maximiza Recall con
   restricción de Precision mínima para evitar spam masivo

SPLIT: Temporal (no aleatorio). Todo antes del cutoff → train.
Datos después del cutoff → test. Esto emula el escenario real de producción.
"""

import json
import hashlib
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, recall_score, precision_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

SEED           = CFG["project"]["random_seed"]
MODELS_DIR     = CFG["paths"]["models"]
MANIFEST       = CFG["paths"]["manifest"]
PROCESSED      = CFG["paths"]["processed_data"]
CUTOFF_DATE    = CFG["temporal"]["cutoff_date"]
TEST_START     = CFG["temporal"]["test_start_date"]
NUM_FEATURES   = CFG["features"]["numeric"]
CAT_FEATURES   = CFG["features"]["categorical"]
MODEL_PARAMS   = CFG["model"]["hyperparams"]
SMOTE_CFG      = CFG["model"]["smote"]
THRESHOLD_CFG  = CFG["model"]["threshold"]


# ─── Preprocesador ────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer que:
    - Escala features numéricas (StandardScaler) → XGBoost es invariante a escala
      pero lo dejamos para consistencia con otros modelos en el ensemble futuro
    - One-Hot Encode variables categóricas (handle_unknown='ignore' para producción)
    """
    numeric_transformer  = StandardScaler()
    # handle_unknown='ignore': Si en producción llega una categoría nueva, no rompe
    # sparse_output=False: XGBoost no acepta matrices sparse directamente
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ],
        remainder="drop",  # Eliminar columnas no especificadas (seguridad)
    )
    return preprocessor


def build_model(scale_pos_weight: float = 1.0) -> XGBClassifier:
    """
    XGBoostClassifier configurado para maximizar Recall.

    HIPERPARÁMETROS CLAVE:
    - scale_pos_weight = n_neg / n_pos: Penaliza errores en clase positiva
      (backup al SMOTE, ayuda cuando SMOTE no es suficiente)
    - eval_metric = 'aucpr': AUC-PR es más informativo que AUC-ROC en
      datasets desbalanceados (no se ve afectado por True Negatives masivos)
    - early_stopping_rounds: Evita overfitting sin necesidad de tuning manual
    """
    params = {**MODEL_PARAMS}
    params["scale_pos_weight"] = scale_pos_weight
    params["random_state"]     = SEED

    return XGBClassifier(**params)


# ─── Ajuste de Umbral Óptimo ──────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float    = THRESHOLD_CFG["min_recall"],
    min_precision: float = THRESHOLD_CFG["min_precision"],
    grid_start: float    = THRESHOLD_CFG["grid_start"],
    grid_end: float      = THRESHOLD_CFG["grid_end"],
    grid_step: float     = THRESHOLD_CFG["grid_step"],
) -> Tuple[float, dict]:
    """
    Grid search sobre el umbral de decisión.
    Busca el umbral que MAXIMIZA Recall, sujeto a:
    - Recall >= min_recall
    - Precision >= min_precision

    RESULTADO: Si ningún umbral cumple ambas restricciones,
    se devuelve el que maximiza Recall (sin restricción de Precision).
    Esto prioriza no perder churners.
    """
    thresholds = np.arange(grid_start, grid_end, grid_step)
    best_threshold = 0.5
    best_recall    = 0.0
    results        = []

    for t in thresholds:
        y_pred  = (y_proba >= t).astype(int)
        rec     = recall_score(y_true, y_pred, zero_division=0)
        prec    = precision_score(y_true, y_pred, zero_division=0)
        results.append({"threshold": round(t, 3), "recall": rec, "precision": prec})

        if rec >= min_recall and prec >= min_precision:
            if rec > best_recall:
                best_recall    = rec
                best_threshold = round(t, 3)

    # Si no se encontró solución factible, maximizar solo Recall
    if best_recall == 0.0:
        logger.warning(
            "⚠️  No se encontró umbral que cumpla ambas restricciones. "
            "Maximizando solo Recall (sin restricción de Precision)."
        )
        best_row       = max(results, key=lambda r: r["recall"])
        best_threshold = best_row["threshold"]
        best_recall    = best_row["recall"]

    logger.info(f"🎯 Umbral óptimo: {best_threshold} | Recall: {best_recall:.3f}")
    return best_threshold, {"threshold_grid": results}


# ─── Train / Test Split Temporal ─────────────────────────────────────────────

def temporal_split(
    feat_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split temporal: todo dato generado antes del cutoff → train,
    después → test.

    NOTA: La feature matrix no tiene una columna de fecha explícita porque
    todas las features son point-in-time al cutoff. Por eso, el split
    se hace usando el índice (customer_id) que mapeamos a los labels,
    donde sabemos si un cliente pertenece a la cohorte de test o no.

    En este setup simplificado: usamos una proporción 80/20 porque todos
    los clientes tienen el mismo cutoff. En producción con múltiples cohortes,
    el split sería por fecha de cutoff.
    """
    y = feat_matrix["churn_label"]
    X = feat_matrix.drop(columns=["churn_label"])

    # Para reproducibilidad, ordena por customer_id (determinista)
    X = X.sort_index()
    y = y.loc[X.index]

    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(
        f"📊 Split temporal: Train={len(X_train):,} | Test={len(X_test):,} | "
        f"Churn en train={y_train.mean():.1%} | Churn en test={y_test.mean():.1%}"
    )
    return X_train, X_test, y_train, y_test


# ─── Pipeline de Entrenamiento ────────────────────────────────────────────────

def train(feat_matrix: Optional[pd.DataFrame] = None) -> dict:
    """
    Entrenamiento completo con:
    1. Split temporal 80/20
    2. Preprocesamiento → SMOTE → XGBoost (pipeline imblearn)
    3. Ajuste de umbral para maximizar Recall ≥ 0.80

    SMOTE se aplica DENTRO del pipeline, lo que garantiza que
    las muestras sintéticas se generan SOLO con datos de train
    y después de la transformación numérica.
    """
    logger.info("🤖 Iniciando entrenamiento del modelo de churn...")

    # ── Cargar datos si no se pasan ────────────────────────────────────────────
    if feat_matrix is None:
        parquet_path = Path(PROCESSED) / "feature_matrix.parquet"
        logger.info(f"Cargando feature matrix desde {parquet_path}...")
        feat_matrix = pd.read_parquet(parquet_path)

    # ── Split ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = temporal_split(feat_matrix)

    # ── Calcular scale_pos_weight ─────────────────────────────────────────────
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"⚖️  scale_pos_weight = {scale_pos_weight:.2f} (n_neg={n_neg}, n_pos={n_pos})")

    # ── Construir Pipeline ────────────────────────────────────────────────────
    # imblearn.Pipeline maneja SMOTE correctamente (no se aplica en predict())
    preprocessor = build_preprocessor()
    smote        = SMOTE(
        sampling_strategy=SMOTE_CFG["sampling_strategy"],
        k_neighbors=SMOTE_CFG["k_neighbors"],
        random_state=SMOTE_CFG["random_state"],
    )
    classifier = build_model(scale_pos_weight=scale_pos_weight)

    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        # SMOTE va después del preprocesador porque necesita datos numéricos continuos
        # (no puede generar sintéticos en espacio categórico crudo)
        ("smote",        smote),
        ("classifier",   classifier),
    ])

    # ── Entrenar ──────────────────────────────────────────────────────────────
    logger.info("⏳ Entrenando pipeline (PreProcessor → SMOTE → XGBoost)...")
    pipeline.fit(X_train, y_train)
    logger.info("✅ Entrenamiento completado.")

    # ── Probabilidades de predicción ──────────────────────────────────────────
    # Usamos predict_proba en lugar de predict porque necesitamos ajustar el umbral
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    y_proba_test  = pipeline.predict_proba(X_test)[:, 1]

    # ── Ajuste de Umbral ──────────────────────────────────────────────────────
    optimal_threshold, threshold_details = find_optimal_threshold(
        y_true  = y_train.values,
        y_proba = y_proba_train,
    )

    # Predicciones finales con umbral ajustado
    y_pred_test = (y_proba_test >= optimal_threshold).astype(int)

    # ── Métricas en Test Set ──────────────────────────────────────────────────
    test_recall    = recall_score(y_test, y_pred_test, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_auc_roc   = roc_auc_score(y_test, y_proba_test)
    test_auc_pr    = average_precision_score(y_test, y_proba_test)
    clf_report     = classification_report(y_test, y_pred_test, output_dict=True)

    logger.info(
        f"\n{'='*50}\n"
        f"📈 Métricas en TEST SET (umbral={optimal_threshold}):\n"
        f"   Recall (Churn=1): {test_recall:.4f}\n"
        f"   Precision:        {test_precision:.4f}\n"
        f"   AUC-ROC:          {test_auc_roc:.4f}\n"
        f"   AUC-PR:           {test_auc_pr:.4f}\n"
        f"{'='*50}"
    )

    # ── Serializar modelo ─────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path    = Path(MODELS_DIR) / f"churn_model_{model_version}.pkl"

    model_artifact = {
        "pipeline":          pipeline,
        "optimal_threshold": optimal_threshold,
        "model_version":     model_version,
        "num_features":      NUM_FEATURES,
        "cat_features":      CAT_FEATURES,
        "train_metrics": {
            "n_train": len(X_train),
            "n_pos":   n_pos,
            "n_neg":   n_neg,
            "churn_rate_train": float(y_train.mean()),
        },
        "test_metrics": {
            "recall":    float(test_recall),
            "precision": float(test_precision),
            "auc_roc":   float(test_auc_roc),
            "auc_pr":    float(test_auc_pr),
            "report":    clf_report,
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    # Hash MD5 del modelo para versionado
    with open(model_path, "rb") as f:
        model_hash = hashlib.md5(f.read()).hexdigest()

    logger.info(f"💾 Modelo guardado: {model_path} | MD5: {model_hash[:8]}...")

    # ── Guardar en manifest ───────────────────────────────────────────────────
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)

    manifest["model"] = {
        "path":          str(model_path),
        "version":       model_version,
        "md5":           model_hash,
        "threshold":     optimal_threshold,
        "test_recall":   float(test_recall),
        "test_auc_roc":  float(test_auc_roc),
        "cutoff_date":   CUTOFF_DATE,
        "updated_at":    datetime.utcnow().isoformat(),
    }

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "model_path":    str(model_path),
        "model_version": model_version,
        "threshold":     optimal_threshold,
        "metrics":       {"recall": test_recall, "precision": test_precision,
                          "auc_roc": test_auc_roc, "auc_pr": test_auc_pr},
        "y_test":        y_test,
        "y_proba_test":  y_proba_test,
        "y_pred_test":   y_pred_test,
        "X_test":        X_test,
    }


if __name__ == "__main__":
    result = train()
    logger.info(f"✅ Modelo listo: {result['model_path']}")
