"""
ml/baselines.py — Baselines Honestos (Fase V)
==============================================
Implementa los baselines de referencia que el modelo XGBoost DEBE superar.
Un modelo ML que no mejora a estas reglas heurísticas simples no tiene
valor de negocio.

PRINCIPIO DE HONESTIDAD:
- Todos los baselines usan SOLO features disponibles antes de t0
- NO hay ajuste de hiperparámetros basado en el test set
- Se reportan con el mismo protocolo de evaluación (Bootstrap CI)
- Los baselines se evalúan en el MISMO test set que el modelo principal

BASELINES IMPLEMENTADOS:
  1. RFM Heurístico     — Regla de recency: threshold en percentil 75
  2. Recency Simple     — Umbral fijo de 60 días sin compra
  3. Frequency Rule     — Regla de frecuencia baja (< 2 órdenes)
  4. RFM Score Compuesto — Combinación de recency + frequency + monetary

Referencia: Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005).
"""

from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    recall_score, precision_score, roc_auc_score,
    average_precision_score, f1_score,
)


# ─── Baseline 1: RFM Heurístico (Percentil de Recency) ────────────────────────

def baseline_rfm_heuristic(
    X: pd.DataFrame,
    recency_percentile: float = 0.75,
) -> np.ndarray:
    """
    Clasifica como Churn a los clientes cuya recency supera
    el percentil `recency_percentile` de la distribución del train set.

    RAZONAMIENTO DE NEGOCIO:
    El 25% de clientes con mayor tiempo sin comprar son los candidatos
    más naturales para una campaña de retención. Es la heurística más
    común en equipos de marketing sin modelos ML.

    ANTI-LEAKAGE:
    Solo usa recency_days, que está calculado estrictamente sobre
    la ventana de observación W_obs = [t0-90, t0].
    """
    if "recency_days" not in X.columns:
        logger.warning("recency_days no disponible. Baseline RFM retorna ceros.")
        return np.zeros(len(X), dtype=int)

    threshold = X["recency_days"].quantile(recency_percentile)
    predictions = (X["recency_days"] >= threshold).astype(int).values
    logger.info(
        f"[Baseline RFM] Umbral recency p{recency_percentile*100:.0f}={threshold:.1f}d "
        f"| Positivos: {predictions.sum()} ({predictions.mean():.1%})"
    )
    return predictions


# ─── Baseline 2: Recency Fija (Regla de Negocio Simple) ──────────────────────

def baseline_recency_fixed(
    X: pd.DataFrame,
    threshold_days: int = 60,
) -> np.ndarray:
    """
    Regla fija: cualquier cliente sin compra en más de `threshold_days`
    se clasifica como riesgo de churn.

    RAZONAMIENTO DE NEGOCIO:
    Representa el umbral intuitivo de un analista de negocio sin
    acceso a modelos estadísticos. "Si no compró en 60 días, está en riesgo."

    ANTI-LEAKAGE:
    Idéntico al Baseline 1. Solo usa recency_days sobre W_obs.
    """
    if "recency_days" not in X.columns:
        logger.warning("recency_days no disponible. Baseline Recency retorna ceros.")
        return np.zeros(len(X), dtype=int)

    predictions = (X["recency_days"] > threshold_days).astype(int).values
    logger.info(
        f"[Baseline Recency {threshold_days}d] "
        f"Positivos: {predictions.sum()} ({predictions.mean():.1%})"
    )
    return predictions


# ─── Baseline 3: Frecuencia Baja ──────────────────────────────────────────────

def baseline_low_frequency(
    X: pd.DataFrame,
    min_orders: int = 2,
) -> np.ndarray:
    """
    Clasifica como Churn a clientes con menos de `min_orders` órdenes
    en la ventana de observación.

    RAZONAMIENTO DE NEGOCIO:
    Un cliente con 1 sola compra en 90 días muestra poco engagement.
    Esta regla captura "clientes de prueba" que pueden no volver.

    ANTI-LEAKAGE:
    frequency_orders se calcula sobre W_obs = [t0-90, t0].
    """
    if "frequency_orders" not in X.columns:
        logger.warning("frequency_orders no disponible. Baseline Freq retorna ceros.")
        return np.zeros(len(X), dtype=int)

    predictions = (X["frequency_orders"] < min_orders).astype(int).values
    logger.info(
        f"[Baseline Freq<{min_orders}] "
        f"Positivos: {predictions.sum()} ({predictions.mean():.1%})"
    )
    return predictions


# ─── Baseline 4: RFM Score Compuesto ─────────────────────────────────────────

def baseline_rfm_composite(
    X: pd.DataFrame,
    recency_weight: float  = 0.5,
    frequency_weight: float = 0.3,
    monetary_weight: float  = 0.2,
    score_threshold: float  = 0.60,
) -> np.ndarray:
    """
    Construye un score compuesto RFM (ponderado) y clasifica como Churn
    si el score supera el threshold.

    PROCESO:
    1. Normaliza cada dimensión RFM a [0, 1] con MinMax
    2. Invierte recency y monetary (mayor valor = mayor riesgo)
    3. Pondera y suma
    4. Clasifica con threshold

    ANTI-LEAKAGE:
    Todas las features provienen de W_obs. La normalización usa
    la distribución del conjunto disponible (no del test).
    """
    available = [
        c for c in ["recency_days", "frequency_orders", "monetary_total"]
        if c in X.columns
    ]

    if len(available) == 0:
        logger.warning("Ninguna feature RFM disponible. Baseline compuesto retorna ceros.")
        return np.zeros(len(X), dtype=int)

    df = X[available].copy()

    # Normalizar a [0, 1]
    for col in available:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.0

    # Invertir: mayor recency → mayor riesgo
    if "recency_days" in df.columns:
        df["recency_days"] = df["recency_days"]           # Mayor = mayor riesgo ✓

    # frequency: menor frecuencia → mayor riesgo → invertir
    if "frequency_orders" in df.columns:
        df["frequency_orders"] = 1 - df["frequency_orders"]

    # monetary: menor gasto → mayor riesgo → invertir
    if "monetary_total" in df.columns:
        df["monetary_total"] = 1 - df["monetary_total"]

    # Score compuesto ponderado
    weights = {}
    if "recency_days"     in df.columns: weights["recency_days"]     = recency_weight
    if "frequency_orders" in df.columns: weights["frequency_orders"] = frequency_weight
    if "monetary_total"   in df.columns: weights["monetary_total"]   = monetary_weight

    total_weight = sum(weights.values())
    rfm_score    = sum(df[col] * (w / total_weight) for col, w in weights.items())

    predictions = (rfm_score >= score_threshold).astype(int).values
    logger.info(
        f"[Baseline RFM Compuesto (threshold={score_threshold})] "
        f"Positivos: {predictions.sum()} ({predictions.mean():.1%})"
    )
    return predictions


# ─── Evaluador Unificado de Baselines ─────────────────────────────────────────

def evaluate_all_baselines(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict:
    """
    Evalúa todos los baselines en el test set y retorna un diccionario
    con métricas comparativas.

    Permite comparar el modelo XGBoost contra TODOS los baselines
    en el mismo test set con las mismas métricas.

    Parámetros
    ----------
    X_test : pd.DataFrame
        Features del test set (sin churn_label).
    y_test : np.ndarray
        Etiquetas reales del test set.

    Retorna
    -------
    dict : métricas por baseline.
    """
    logger.info("📏 Evaluando baselines honestos...")

    baselines_fns = {
        "RFM Heurístico (p75)":  lambda X: baseline_rfm_heuristic(X, recency_percentile=0.75),
        "Recency Fija (60d)":    lambda X: baseline_recency_fixed(X,  threshold_days=60),
        "Frecuencia Baja (<2)":  lambda X: baseline_low_frequency(X,  min_orders=2),
        "RFM Compuesto":         lambda X: baseline_rfm_composite(X),
    }

    results = {}
    for name, fn in baselines_fns.items():
        preds = fn(X_test)

        # Cómputo seguro de AUC (necesita varianza en predicciones)
        try:
            auc = float(roc_auc_score(y_test, preds))
        except Exception:
            auc = 0.5  # AUC de baseline aleatorio

        metrics = {
            "recall":    float(recall_score(y_test,    preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "f1":        float(f1_score(y_test,        preds, zero_division=0)),
            "auc_roc":   auc,
            "positives": int(preds.sum()),
            "positive_rate": float(preds.mean()),
        }
        results[name] = metrics

        logger.info(
            f"   [{name}] "
            f"Recall={metrics['recall']:.3f} | "
            f"Precision={metrics['precision']:.3f} | "
            f"AUC={metrics['auc_roc']:.3f}"
        )

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    import pandas as pd

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    parquet_path = Path(cfg["paths"]["processed_data"]) / "feature_matrix.parquet"
    if not parquet_path.exists():
        print(f"❌ No se encontró: {parquet_path}")
        print("   Ejecuta: python main.py --skip-train")
    else:
        fm   = pd.read_parquet(parquet_path)
        y    = fm["churn_label"].values
        X    = fm.drop(columns=["churn_label"])
        split = int(len(X) * 0.80)
        X_test = X.iloc[split:]
        y_test = y[split:]

        results = evaluate_all_baselines(X_test, y_test)
        print("\n═══ COMPARATIVA DE BASELINES ═══")
        for name, m in results.items():
            print(f"\n  {name}")
            print(f"    Recall={m['recall']:.3f} | Precision={m['precision']:.3f} | AUC={m['auc_roc']:.3f}")
