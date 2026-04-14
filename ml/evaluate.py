"""
ml/evaluate.py — Evaluación Rigurosa del Modelo
================================================
Implementa:
1. Métricas completas en test set
2. Bootstrap CI para el Recall (1000 iteraciones)
3. Baseline honestos (regla RFM simple) para comparación justa
4. Generación de figuras para el reporte LaTeX

DECISIÓN: El Bootstrap CI es esencial en producción para comunicar
no solo el Recall puntual sino la incertidumbre alrededor de ese valor.
Un Recall de 0.82 ± 0.06 (95% CI) es información muy diferente a uno de
0.82 ± 0.02.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend sin GUI para entornos de servidor
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from loguru import logger
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
    recall_score, precision_score, f1_score,
)


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FIGURES_DIR       = CFG["paths"]["figures"]
BOOT_ITERS        = CFG["evaluation"]["bootstrap_iterations"]
BOOT_ALPHA        = CFG["evaluation"]["bootstrap_alpha"]
SEED              = CFG["project"]["random_seed"]

# Estilo visual unificado para el reporte
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = ["#2D6A4F", "#E63946"]   # Verde=No Churn, Rojo=Churn


# ─── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_recall_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = BOOT_ITERS,
    alpha: float = BOOT_ALPHA,
    random_seed: int = SEED,
) -> dict[str, float]:
    """
    Calcula el intervalo de confianza del Recall usando Bootstrap no paramétrico.
    No asume distribución normal → más robusto para métricas de clasificación.

    PROCESO:
    1. Resamplea con reemplazo n_iterations veces
    2. Calcula Recall en cada muestra
    3. Toma percentiles alpha/2 y 1-alpha/2 como límites del CI
    """
    rng     = np.random.default_rng(random_seed)
    recalls = []
    n       = len(y_true)

    for _ in range(n_iterations):
        idx     = rng.integers(0, n, size=n)
        r       = recall_score(y_true[idx], y_pred[idx], zero_division=0)
        recalls.append(r)

    recalls     = np.array(recalls)
    lo, hi      = np.percentile(recalls, [alpha/2 * 100, (1 - alpha/2) * 100])
    point_est   = float(np.mean(recalls))

    return {
        "recall_point":  point_est,
        "recall_ci_low": float(lo),
        "recall_ci_high": float(hi),
        "ci_level": f"{int((1 - alpha) * 100)}%",
    }


# ─── Baseline Honesto: Regla RFM ─────────────────────────────────────────────

def rfm_baseline(X_test: pd.DataFrame, recency_threshold: int = 60) -> np.ndarray:
    """
    Baseline heurístico: Si recency_days > threshold → predice churn.
    Este es el benchmark mínimo que el modelo ML DEBE superar.
    Si el XGBoost no mejora a esta regla simple, el modelo es inútil.
    """
    if "recency_days" not in X_test.columns:
        logger.warning("recency_days no encontrado en X_test. Baseline retorna ceros.")
        return np.zeros(len(X_test), dtype=int)
    return (X_test["recency_days"] > recency_threshold).astype(int).values


# ─── Figuras para el Reporte ──────────────────────────────────────────────────

def _plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> None:
    """Genera curva ROC y Precision-Recall en una sola figura."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ─ ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_roc     = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color="#E63946", lw=2, label=f"XGBoost (AUC={auc_roc:.3f})")
    axes[0].plot([0,1], [0,1], "k--", lw=1, label="Random baseline")
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="Curva ROC — Churn Predictor")
    axes[0].legend()

    # ─ Precision-Recall Curve ────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    auc_pr       = average_precision_score(y_true, y_proba)
    axes[1].plot(rec, prec, color="#2D6A4F", lw=2, label=f"XGBoost (AP={auc_pr:.3f})")
    axes[1].axhline(y_true.mean(), color="gray", linestyle="--", lw=1,
                    label=f"Baseline (prevalencia={y_true.mean():.2f})")
    axes[1].set(xlabel="Recall", ylabel="Precision",
                title="Curva Precision-Recall — Churn Predictor")
    axes[1].legend()

    plt.tight_layout()
    path = Path(FIGURES_DIR) / "roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Figura guardada: {path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> None:
    """Matriz de confusión con anotaciones de negocio."""
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["No Churn\n(Retain)", "Churn\n(Risk)"]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn_r",
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set(
        xlabel="Predicción del Modelo",
        ylabel="Valor Real",
        title=f"Matriz de Confusión (umbral={threshold})",
    )
    # Anotar tipo de error
    ax.text(0.5, 1.5, "FP: Cupón\ninnecesario", ha="center", va="center",
            fontsize=9, color="#555")
    ax.text(1.5, 0.5, "FN: Cliente\nperdido ⚠️", ha="center", va="center",
            fontsize=9, color="#c00", weight="bold")

    plt.tight_layout()
    path = Path(FIGURES_DIR) / "confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"📊 Figura guardada: {path}")


def _plot_feature_importance(pipeline, top_n: int = 15) -> None:
    """Top N features más importantes según XGBoost (gain)."""
    try:
        clf = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]

        # Reconstruye nombres de features post-transformación
        # handle spending_intensity addition
        num_names = CFG["features"]["numeric"]
        cat_names = list(
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(CFG["features"]["categorical"])
        )
        all_names = num_names + cat_names

        importances = clf.feature_importances_
        if len(importances) != len(all_names):
            logger.warning(f"Feature mismatch: {len(importances)} != {len(all_names)}. Using indices.")
            all_names = [f"feat_{i}" for i in range(len(importances))]

        feat_df = pd.DataFrame({"feature": all_names, "importance": importances})
        feat_df = feat_df.nlargest(top_n, "importance")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feat_df["feature"], feat_df["importance"], color="#2D6A4F")
        ax.set(xlabel="Importancia (Gain)", title=f"Top {top_n} Features — XGBoost")
        ax.invert_yaxis()
        plt.tight_layout()
        path = Path(FIGURES_DIR) / "feature_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 Figura guardada: {path}")
    except Exception as e:
        logger.warning(f"No se pudo generar feature importance: {e}")


# ─── Business Analytics (Lift / Recall@K) ───────────────────────────────────

def compute_lift_and_recall_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k_percent: float = 0.20,
) -> dict:
    """
    Calcula Lift y Recall en el top K% de la población (ordenada por riesgo).
    Esencial para dimensionar el costo de la campaña de retención.
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False)
    
    n_total = len(df)
    n_k     = int(n_total * k_percent)
    
    top_k   = df.head(n_k)
    
    # Recall @ K: ¿Qué % del total de churners capturamos en el top K%?
    total_churners = df["y_true"].sum()
    captured       = top_k["y_true"].sum()
    recall_at_k    = captured / total_churners if total_churners > 0 else 0.0
    
    # Lift @ K: ¿Cuántas veces mejor es el modelo vs aleatorio en el top K%?
    avg_churn_rate = total_churners / n_total
    k_churn_rate   = captured / n_k if n_k > 0 else 0.0
    lift_at_k      = k_churn_rate / avg_churn_rate if avg_churn_rate > 0 else 1.0
    
    return {
        "k_percent":   k_percent,
        "recall_at_k": float(recall_at_k),
        "lift_at_k":   float(lift_at_k),
        "captured":    int(captured),
        "total":       int(total_churners),
    }


def _plot_lift_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Genera curva de Lift acumulado."""
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba})
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    
    df["cumulative_true"] = df["y_true"].cumsum()
    df["cumulative_pop"]  = np.arange(1, len(df) + 1) / len(df)
    df["cumulative_recall"] = df["cumulative_true"] / df["y_true"].sum()
    
    avg_rate = df["y_true"].sum() / len(df)
    df["cumulative_lift"] = (df["cumulative_true"] / (np.arange(1, len(df) + 1))) / avg_rate
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["cumulative_pop"], df["cumulative_lift"], color="#219EBC", lw=2, label="Curva de Lift")
    ax.axhline(1, color="red", linestyle="--", label="Baseline (Aleatorio)")
    ax.set(xlabel="% de Población (Top Riesgo)", ylabel="Lift", title="Curva de Lift Acumulado")
    ax.legend()
    
    plt.tight_layout()
    path = Path(FIGURES_DIR) / "lift_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Evaluación Completa ──────────────────────────────────────────────────────

def full_evaluation(
    pipeline,
    X_test:    pd.DataFrame,
    y_test:    pd.Series,
    y_proba:   np.ndarray,
    threshold: float,
) -> dict:
    """
    Evaluación completa del modelo y generación de todos los artefactos.
    Retorna dict con todas las métricas para el reporte LaTeX.
    """
    y_pred = (y_proba >= threshold).astype(int)
    y_true = y_test.values

    # ─ Baseline ───────────────────────────────────────────────────────────────
    baseline_pred = rfm_baseline(X_test)
    baseline_recall   = recall_score(y_true, baseline_pred, zero_division=0)
    baseline_prec     = precision_score(y_true, baseline_pred, zero_division=0)
    logger.info(
        f"📏 Baseline RFM: Recall={baseline_recall:.3f} | Precision={baseline_prec:.3f}"
    )

    # ─ Bootstrap CI ───────────────────────────────────────────────────────────
    ci_results = bootstrap_recall_ci(y_true, y_pred)
    logger.info(
        f"📊 Bootstrap CI Recall (1000 iter): "
        f"{ci_results['recall_point']:.3f} "
        f"[{ci_results['recall_ci_low']:.3f}, {ci_results['recall_ci_high']:.3f}] "
        f"({ci_results['ci_level']} CI)"
    )

    # ─ Business Metrics ───────────────────────────────────────────────────────
    lift_metrics = compute_lift_and_recall_at_k(y_true, y_proba, k_percent=0.20)
    logger.info(
        f"🚀 Business Metric: Lift@20%={lift_metrics['lift_at_k']:.2f} | "
        f"Recall@20%={lift_metrics['recall_at_k']:.1%}"
    )

    # ─ Métricas finales ───────────────────────────────────────────────────────
    metrics = {
        "model": {
            "threshold":  threshold,
            "recall":     float(recall_score(y_true, y_pred, zero_division=0)),
            "precision":  float(precision_score(y_true, y_pred, zero_division=0)),
            "f1":         float(f1_score(y_true, y_pred, zero_division=0)),
            "auc_roc":    float(roc_auc_score(y_true, y_proba)),
            "auc_pr":     float(average_precision_score(y_true, y_proba)),
        },
        "business": lift_metrics,
        "bootstrap_ci": ci_results,
        "baseline_rfm": {
            "recall":    float(baseline_recall),
            "precision": float(baseline_prec),
        },
        "test_set_size": len(y_true),
        "churn_rate_test": float(y_true.mean()),
    }

    # ─ Figuras ─────────────────────────────────────────────────────────────────
    _plot_roc_pr_curves(y_true, y_proba, threshold)
    _plot_confusion_matrix(y_true, y_pred, threshold)
    _plot_feature_importance(pipeline)
    _plot_lift_curve(y_true, y_proba)

    return metrics
