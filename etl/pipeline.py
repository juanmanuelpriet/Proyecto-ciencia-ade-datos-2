"""
etl/pipeline.py — Orquestador DAG del ETL
==========================================
Define el grafo de transformación completo y lo ejecuta en orden topológico.
Cada nodo del grafo registra su output en el manifest de artefactos.
"""

import json
import hashlib
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from etl.extractor import RawDataExtractor
from etl.cleaner import clean_orders, clean_support_tickets, clean_customers
from etl.feature_engineer import build_feature_matrix


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

CUTOFF_DATE  = CFG["temporal"]["cutoff_date"]
OBS_WINDOW   = CFG["temporal"]["observation_window_days"]
PROCESSED    = CFG["paths"]["processed_data"]
MANIFEST     = CFG["paths"]["manifest"]
DB_PATH      = CFG["paths"]["db"]


def _update_manifest(key: str, value: dict) -> None:
    """Registra el artefacto generado en manifest.json (linaje de datos)."""
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)
    manifest[key] = {**value, "updated_at": datetime.utcnow().isoformat()}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def _df_hash(df: pd.DataFrame) -> str:
    """Hash MD5 de un DataFrame para verificar reproducibilidad."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


# ─── Nodos del DAG ────────────────────────────────────────────────────────────

def node_extract(cutoff_date: str = CUTOFF_DATE) -> dict[str, pd.DataFrame]:
    """NODE 1: Extracción de datos brutos."""
    logger.info("📦 [DAG Node 1/4] Extracción de datos brutos...")
    with RawDataExtractor(db_path=DB_PATH) as ext:
        raw = ext.get_full_dataset_for_feature_engineering(
            cutoff_date=cutoff_date,
            observation_window_days=OBS_WINDOW,
        )
    logger.info(f"   ✅ Órdenes: {len(raw['orders']):,} | Clientes: {len(raw['customers']):,}")
    return raw


def node_clean(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """NODE 2: Limpieza y validación de invariantes."""
    logger.info("🧹 [DAG Node 2/4] Limpieza y validación...")

    orders_clean, orders_report    = clean_orders(raw["orders"])
    tickets_clean, tickets_report  = clean_support_tickets(raw["support_tickets"])
    customers_clean, cust_report   = clean_customers(raw["customers"])

    # Log de invariantes fallidas
    for report in [orders_report, tickets_report, cust_report]:
        for inv in report.invariant_results:
            status = "✅ PASS" if inv.passed else "❌ FAIL"
            logger.info(f"   Invariante [{inv.name}]: {status} ({inv.n_violations} violaciones)")

    return {
        "customers":      customers_clean,
        "orders":         orders_clean,
        "order_items":    raw["order_items"],
        "support_tickets": tickets_clean,
        "churn_labels":   raw["churn_labels"],
    }


def node_feature_engineering(clean_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """NODE 3: Feature engineering point-in-time correcto."""
    logger.info("⚙️  [DAG Node 3/4] Feature engineering...")
    from datetime import date
    cutoff = date.fromisoformat(CUTOFF_DATE)

    feat_matrix = build_feature_matrix(
        customers=clean_data["customers"],
        orders=clean_data["orders"],
        order_items=clean_data["order_items"],
        tickets=clean_data["support_tickets"],
        labels=clean_data["churn_labels"],
        cutoff=cutoff,
        window_days=OBS_WINDOW,
    )
    return feat_matrix


def node_save(feat_matrix: pd.DataFrame) -> Path:
    """NODE 4: Persistencia del dataset procesado."""
    logger.info("💾 [DAG Node 4/4] Guardando dataset procesado...")
    os.makedirs(PROCESSED, exist_ok=True)

    output_path = Path(PROCESSED) / "feature_matrix.parquet"
    feat_matrix.to_parquet(output_path, index=True)

    data_hash = _df_hash(feat_matrix)
    _update_manifest("feature_matrix", {
        "path":         str(output_path),
        "rows":         len(feat_matrix),
        "columns":      list(feat_matrix.columns),
        "md5":          data_hash,
        "churn_rate":   float(feat_matrix["churn_label"].mean()),
        "cutoff_date":  CUTOFF_DATE,
    })

    logger.info(f"✅ Dataset guardado: {output_path} | Hash: {data_hash[:8]}...")
    return output_path


# ─── DAG Runner ───────────────────────────────────────────────────────────────

def run_etl_pipeline() -> pd.DataFrame:
    """
    Ejecuta el pipeline ETL completo en orden topológico (DAG).
    Retorna la feature matrix lista para el modelo.
    """
    logger.info("🚀 Iniciando ETL Pipeline...")
    start = datetime.utcnow()

    raw         = node_extract()
    clean_data  = node_clean(raw)
    feat_matrix = node_feature_engineering(clean_data)
    node_save(feat_matrix)

    elapsed = (datetime.utcnow() - start).total_seconds()
    logger.info(f"🏁 Pipeline completado en {elapsed:.1f}s")

    return feat_matrix


if __name__ == "__main__":
    df = run_etl_pipeline()
    print(df.head())
    print(df.describe())
