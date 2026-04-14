"""
main.py — Orquestador Maestro del Sistema de Churn
====================================================
Ejecuta el pipeline completo en orden:
  1. Seed de datos (idempotente)
  2. ETL + Feature Engineering
  3. EDA Report
  4. Entrenamiento del modelo
  5. Evaluación completa
  6. Actualización del manifest

Uso:
  python main.py
  python main.py --skip-seed      # Si los datos ya existen
  python main.py --skip-train     # Solo regenera EDA y métricas
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import yaml
from loguru import logger

# Configurar logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "artifacts/pipeline.log",
    rotation="10 MB",
    level="DEBUG",
)


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       🛒 Global-E-Shop Churn Prediction System v1.0.0        ║
║          Sistema End-to-End de Retención de Clientes         ║
╚══════════════════════════════════════════════════════════════╝
    """)


def parse_args():
    p = argparse.ArgumentParser(description="Orquestador del pipeline de Churn")
    p.add_argument("--skip-seed",  action="store_true", help="Omitir generación de datos")
    p.add_argument("--skip-eda",   action="store_true", help="Omitir EDA report")
    p.add_argument("--skip-train", action="store_true", help="Omitir entrenamiento")
    p.add_argument("--force-seed", action="store_true", help="Forzar regeneración de datos")
    return p.parse_args()


def main():
    banner()
    args     = parse_args()
    t_start  = time.time()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    os.makedirs("artifacts/data/raw", exist_ok=True)
    os.makedirs("artifacts/data/processed", exist_ok=True)
    os.makedirs("artifacts/models", exist_ok=True)
    os.makedirs("artifacts/figures", exist_ok=True)

    # ── PASO 1: Seed de Datos ──────────────────────────────────────────────────
    if not args.skip_seed:
        logger.info("=" * 60)
        logger.info("PASO 1/5: Generación de datos sintéticos")
        logger.info("=" * 60)
        from sql.seed_data_02 import seed
        seed(force=args.force_seed)
    else:
        logger.info("⏭  PASO 1 omitido (--skip-seed)")

    # ── PASO 2: ETL Pipeline ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PASO 2/5: Pipeline ETL")
    logger.info("=" * 60)
    from etl.pipeline import run_etl_pipeline
    feat_matrix = run_etl_pipeline()

    # ── PASO 3: EDA Report ────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("=" * 60)
        logger.info("PASO 3/5: Reporte de Calidad de Datos (EDA)")
        logger.info("=" * 60)
        from eda.data_quality_report import generate_eda_report
        eda_report = generate_eda_report(feat_matrix)
        logger.info(
            f"   Clientes: {eda_report['n_customers']:,} | "
            f"Churn rate: {eda_report['churn_rate']:.1%} | "
            f"Features con nulos: {len(eda_report['null_report'])}"
        )
    else:
        logger.info("⏭  PASO 3 omitido (--skip-eda)")
        eda_report = {}

    # ── PASO 4: Entrenamiento ─────────────────────────────────────────────────
    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("PASO 4/5: Entrenamiento del Modelo")
        logger.info("=" * 60)
        from ml.train import train
        train_result = train(feat_matrix=feat_matrix)

        # ── PASO 5: Evaluación ────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("PASO 5/5: Evaluación Completa")
        logger.info("=" * 60)
        import pickle
        with open(train_result["model_path"], "rb") as f:
            artifact = pickle.load(f)

        from ml.evaluate import full_evaluation
        eval_metrics = full_evaluation(
            pipeline=artifact["pipeline"],
            X_test=train_result["X_test"],
            y_test=train_result["y_test"],
            y_proba=train_result["y_proba_test"],
            threshold=train_result["threshold"],
        )

        logger.info(
            f"\n{'='*60}\n"
            f"📋 RESUMEN FINAL DEL SISTEMA\n"
            f"{'='*60}\n"
            f"  Modelo:            {train_result['model_version']}\n"
            f"  Umbral:            {train_result['threshold']}\n"
            f"  Recall (test):     {eval_metrics['model']['recall']:.4f}\n"
            f"  Precision (test):  {eval_metrics['model']['precision']:.4f}\n"
            f"  AUC-ROC (test):    {eval_metrics['model']['auc_roc']:.4f}\n"
            f"  AUC-PR (test):     {eval_metrics['model']['auc_pr']:.4f}\n"
            f"  Recall CI 95%%:   [{eval_metrics['bootstrap_ci']['recall_ci_low']:.3f}, "
            f"{eval_metrics['bootstrap_ci']['recall_ci_high']:.3f}]\n"
            f"  Baseline Recall:   {eval_metrics['baseline_rfm']['recall']:.4f}\n"
            f"{'='*60}"
        )

    else:
        logger.info("⏭  PASOs 4-5 omitidos (--skip-train)")
        eval_metrics = {}

    # ── Tiempo total ──────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info(f"🏁 Pipeline completo en {elapsed:.1f}s")
    logger.info("💡 Para iniciar la API: uvicorn api.app:app --host 0.0.0.0 --port 8000")
    logger.info("📚 Documentación: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
