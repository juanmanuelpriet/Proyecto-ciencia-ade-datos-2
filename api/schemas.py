"""
api/schemas.py — Pydantic Schemas para la API de Churn
=======================================================
Validación estricta de tipos y rangos en entrada y salida.
Pydantic v2 con field validators para reglas de negocio.
"""

from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ─── Request ─────────────────────────────────────────────────────────────────

class ChurnPredictionRequest(BaseModel):
    """
    Features de un cliente para predicción de churn.
    Todos los campos numéricos tienen rangos físicamente válidos.
    """
    model_config = ConfigDict(protected_namespaces=())

    customer_id: str = Field(..., description="Identificador único del cliente")

    # RFM
    recency_days:       float = Field(..., ge=0, le=3650,
                                      description="Días desde última compra (0-3650)")
    frequency_orders:   int   = Field(..., ge=0, le=10000,
                                      description="Número de órdenes en ventana de observación")
    monetary_total:     float = Field(..., ge=0,
                                      description="Gasto total en COP/USD en la ventana")
    monetary_avg_order: float = Field(..., ge=0,
                                      description="Gasto promedio por orden")

    # Tendencia
    trend_slope_30d:    float = Field(default=0.0,
                                      description="Pendiente del gasto semanal (puede ser negativa)")

    # Diversidad
    category_diversity: int   = Field(..., ge=0, le=50,
                                      description="Número de categorías distintas compradas")

    # Soporte
    support_tickets_90d:    int   = Field(default=0, ge=0,
                                          description="Tickets de soporte abiertos en 90 días")
    avg_satisfaction_score: Optional[float] = Field(
        default=None, ge=1.0, le=5.0,
        description="NPS/CSAT promedio (1-5). Puede ser nulo."
    )

    # Logística
    avg_delivery_days: Optional[float] = Field(
        default=None, ge=0, le=365,
        description="Días promedio de entrega. Puede ser nulo si no hay entregas."
    )
    return_rate:       float = Field(default=0.0, ge=0.0, le=1.0,
                                     description="Tasa de devoluciones (0.0 a 1.0)")

    # Tenure
    days_since_registration: float = Field(..., ge=0,
                                            description="Días desde el registro del cliente")

    # Categóricas
    preferred_category: str = Field(
        default="Unknown",
        description="Categoría de producto más comprada"
    )
    customer_segment: Literal["VIP", "STANDARD", "NEW", "AT_RISK"] = Field(
        default="STANDARD",
        description="Segmento del cliente"
    )

    @field_validator("avg_satisfaction_score", mode="before")
    @classmethod
    def handle_null_satisfaction(cls, v):
        """Permite None explícito. Se imputará con la mediana del entrenamiento."""
        return v  # Pass-through; la imputación ocurre en el predictor

    # Pydantic v2 Config
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra = {
            "example": {
                "customer_id": "C00123",
                "recency_days": 45.0,
                "frequency_orders": 8,
                "monetary_total": 1250.50,
                "monetary_avg_order": 156.31,
                "trend_slope_30d": -12.5,
                "category_diversity": 3,
                "support_tickets_90d": 1,
                "avg_satisfaction_score": 3.5,
                "avg_delivery_days": 5.2,
                "return_rate": 0.125,
                "days_since_registration": 420.0,
                "preferred_category": "Electronics",
                "customer_segment": "STANDARD",
            }
        }
    )


# ─── Response ─────────────────────────────────────────────────────────────────

class ChurnPredictionResponse(BaseModel):
    """Respuesta de predicción de churn."""
    model_config = ConfigDict(protected_namespaces=())

    customer_id:        str
    churn_probability:  float = Field(..., ge=0.0, le=1.0,
                                      description="Probabilidad de churn (0.0 a 1.0)")
    churn_flag:         int   = Field(..., ge=0, le=1,
                                      description="1=Churn, 0=No Churn (determinado por umbral)")
    decision_threshold: float = Field(..., description="Umbral de decisión usado")
    model_version:      str   = Field(..., description="Versión del modelo que generó la predicción")
    recommended_action: str   = Field(..., description="Acción de retención recomendada")

    @property
    def risk_level(self) -> str:
        if self.churn_probability >= 0.75:
            return "CRITICAL"
        elif self.churn_probability >= 0.50:
            return "HIGH"
        elif self.churn_probability >= 0.25:
            return "MEDIUM"
        return "LOW"


class BatchPredictionRequest(BaseModel):
    """Batch de hasta 1000 clientes para predicción simultánea."""
    model_config = ConfigDict(protected_namespaces=())
    customers: List[ChurnPredictionRequest] = Field(
        ..., min_length=1, max_length=1000,
        description="Lista de clientes para predicción"
    )


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions:  List[ChurnPredictionResponse]
    total:        int
    churn_count:  int
    model_version: str


class ModelInfoResponse(BaseModel):
    """Metadata del modelo cargado en memoria."""
    model_config = ConfigDict(protected_namespaces=())
    model_version:   str
    threshold:       float
    test_recall:     float
    test_auc_roc:    float
    cutoff_date:     str
    num_features:    List[str]
    cat_features:    List[str]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status:        str
    model_loaded:  bool
    model_version: Optional[str] = None
