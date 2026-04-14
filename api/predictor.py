"""
api/predictor.py — Wrapper del Modelo para la API
==================================================
Singleton que carga el modelo UNA sola vez al inicio de la API
y lo mantiene en memoria para todas las peticiones subsecuentes.
"""

import pandas as pd
from typing import Optional, Dict, List
from loguru import logger

from ml.predict import load_latest_model, predict


class ChurnPredictor:
    """Singleton que encapsula el modelo y expone una interfaz limpia para la API."""

    _instance = None
    _model_artifact: Optional[Dict] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Carga el modelo en memoria. Llamar una vez al startup de la API."""
        logger.info("🔄 Cargando modelo en memoria...")
        self._model_artifact = load_latest_model()
        logger.info(
            f"✅ Modelo cargado: v{self._model_artifact['model_version']} | "
            f"Threshold: {self._model_artifact['optimal_threshold']}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model_artifact is not None

    @property
    def version(self) -> Optional[str]:
        if self._model_artifact:
            return self._model_artifact["model_version"]
        return None

    @property
    def info(self) -> dict:
        if not self._model_artifact:
            return {}
        m = self._model_artifact
        return {
            "model_version": m["model_version"],
            "threshold":     m["optimal_threshold"],
            "test_recall":   m["test_metrics"]["recall"],
            "test_auc_roc":  m["test_metrics"]["auc_roc"],
            "cutoff_date":   m["train_metrics"].get("cutoff_date", "N/A"),
            "num_features":  m["num_features"],
            "cat_features":  m["cat_features"],
        }

    def predict_one(self, features_dict: dict) -> dict:
        """Predicción para un solo cliente."""
        customer_id = features_dict.pop("customer_id", "UNKNOWN")
        # Imputa nulos antes de crear el DataFrame
        features_dict.setdefault("avg_satisfaction_score", 3.5)
        features_dict.setdefault("avg_delivery_days", 5.0)

        df   = pd.DataFrame([features_dict])
        df.index = [customer_id]

        result = predict(df, model_artifact=self._model_artifact)
        row    = result.iloc[0]

        # Lógica de acción recomendada según probabilidad
        prob = float(row["churn_probability"])
        if prob >= 0.75:
            action = "IMMEDIATE_CALL_SCHEDULED"
        elif prob >= 0.50:
            action = "PRIORITY_COUPON_SENT"
        elif row["churn_flag"] == 1:
            action = "STANDARD_COUPON_SENT"
        else:
            action = "NO_ACTION_REQUIRED"

        return {
            "customer_id":        customer_id,
            "churn_probability":  prob,
            "churn_flag":         int(row["churn_flag"]),
            "decision_threshold": float(row["decision_threshold"]),
            "model_version":      str(row["model_version"]),
            "recommended_action": action,
        }

    def predict_batch(self, customers: list[dict]) -> list[dict]:
        """Predicción en batch eficiente."""
        results = []
        for customer in customers:
            try:
                results.append(self.predict_one(dict(customer)))
            except Exception as e:
                logger.error(f"Error prediciendo {customer.get('customer_id')}: {e}")
        return results


# Instancia global (Singleton)
predictor = ChurnPredictor()
