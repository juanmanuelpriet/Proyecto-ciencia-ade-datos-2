"""
api/app.py — FastAPI Application — Churn Prediction API
=========================================================
Endpoints:
  GET  /health         → Estado de la API y del modelo
  GET  /model/info     → Metadata del modelo cargado
  POST /predict        → Predicción individual
  POST /predict/batch  → Predicción en batch (hasta 1000 clientes)

DECISIÓN de arquitectura: Se usa el evento lifespan de FastAPI para
cargar el modelo UNA sola vez al inicio, no en cada request.
Esto reduce la latencia de ~200ms (cargar modelo) a <5ms por request.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
from api.predictor import predictor


# ─── Lifespan (startup / shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la API; libera recursos al cerrar."""
    logger.info("🚀 Iniciando Global-E-Shop Churn API...")
    try:
        predictor.load()
        logger.info("✅ Modelo listo para servir predicciones.")
    except FileNotFoundError as e:
        logger.critical(
            f"❌ No se encontró modelo entrenado: {e}. "
            "Ejecuta 'python main.py' primero."
        )
    yield
    logger.info("👋 API apagada correctamente.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Global-E-Shop Churn Prediction API",
    description=(
        "API de predicción de churn para el sistema de retención de clientes. "
        "Provee probabilidades de churn y acciones de retención recomendadas."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Healthcheck de la API",
)
async def health_check():
    """Revisa que la API y el modelo estén operativos."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.version,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Modelo"],
    summary="Metadata del modelo activo",
)
async def model_info():
    """Retorna versión, métricas de validación y umbral del modelo en memoria."""
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no cargado. Verifica los logs de startup.",
        )
    return ModelInfoResponse(**predictor.info)


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Predicción"],
    summary="Predicción individual de churn",
    status_code=status.HTTP_200_OK,
)
async def predict_single(request: ChurnPredictionRequest):
    """
    Genera una predicción de churn para un cliente individual.

    Retorna:
    - **churn_probability**: Probabilidad continua [0.0, 1.0]
    - **churn_flag**: 1 si el cliente está en riesgo (según umbral óptimo)
    - **recommended_action**: Acción de retención sugerida
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible.",
        )
    try:
        features = request.model_dump()
        result   = predictor.predict_one(features)
        return ChurnPredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Error en features: {e}",
        )
    except Exception as e:
        logger.error(f"Error inesperado en /predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor.",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predicción"],
    summary="Predicción en batch (hasta 1000 clientes)",
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Genera predicciones de churn para múltiples clientes simultáneamente.
    Ideal para jobs nocturnos o campañas de retención masivas.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible.",
        )
    try:
        customers_data = [c.model_dump() for c in request.customers]
        results        = predictor.predict_batch(customers_data)
        churn_count    = sum(r["churn_flag"] for r in results)

        return BatchPredictionResponse(
            predictions  = [ChurnPredictionResponse(**r) for r in results],
            total        = len(results),
            churn_count  = churn_count,
            model_version = predictor.version or "unknown",
        )
    except Exception as e:
        logger.error(f"Error en /predict/batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    uvicorn.run(
        "api.app:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=cfg["api"]["reload"],
        log_level="info",
    )
