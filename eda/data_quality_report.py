"""
eda/data_quality_report.py — Reporte Automático de Calidad de Datos
====================================================================
Genera un reporte HTML + estadísticas para el reporte LaTeX.
Incluye: missingness, distribuciones, invariantes, balance de clases.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import Optional
from loguru import logger


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FIGURES_DIR  = CFG["paths"]["figures"]
PROCESSED    = CFG["paths"]["processed_data"]

plt.style.use("seaborn-v0_8-whitegrid")


def generate_eda_report(feat_matrix: Optional[pd.DataFrame] = None) -> dict:
    """
    Genera el reporte completo de EDA y guarda figuras en artifacts/figures/.

    Retorna un dict con estadísticas listas para incrustar en el LaTeX.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if feat_matrix is None:
        parquet_path = Path(PROCESSED) / "feature_matrix.parquet"
        logger.info(f"Cargando feature matrix desde {parquet_path}...")
        feat_matrix = pd.read_parquet(parquet_path)

    logger.info(f"📊 Iniciando EDA Report: {feat_matrix.shape[0]:,} clientes × {feat_matrix.shape[1]} columnas")

    # ── 1. Tabla de Missingness ────────────────────────────────────────────────
    null_report = (
        feat_matrix.isnull().sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "null_count"})
    )
    null_report["null_pct"] = (null_report["null_count"] / len(feat_matrix) * 100).round(2)
    null_report = null_report[null_report["null_count"] > 0].sort_values("null_pct", ascending=False)

    logger.info(f"🔍 Columnas con nulos:\n{null_report.to_string(index=False)}")

    # ── 2. Balance de Clases ──────────────────────────────────────────────────
    churn_counts = feat_matrix["churn_label"].value_counts()
    churn_rate   = feat_matrix["churn_label"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    colors  = ["#2D6A4F", "#E63946"]
    bars    = ax.bar(
        ["No Churn (0)", "Churn (1)"],
        churn_counts.values,
        color=colors,
    )
    for bar, val in zip(bars, churn_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\n({val/len(feat_matrix):.1%})",
                ha="center", fontsize=11, weight="bold")
    ax.set(title="Distribución de Clases — Churn Label", ylabel="Número de Clientes")
    ax.set_ylim(0, max(churn_counts.values) * 1.20)
    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"💾 Figura guardada: class_distribution.png")

    # ── 3. Distribuciones de Features Numéricas ───────────────────────────────
    numeric_features = CFG["features"]["numeric"]
    available_num    = [f for f in numeric_features if f in feat_matrix.columns]

    n_cols = 3
    n_rows = (len(available_num) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, feat in enumerate(available_num):
        ax = axes[i]
        # Histograma separado por clase (No Churn vs Churn)
        for label, color, name in zip([0, 1], ["#2D6A4F", "#E63946"], ["No Churn", "Churn"]):
            subset = feat_matrix[feat_matrix["churn_label"] == label][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, density=True)
        ax.set(title=feat, xlabel=feat, ylabel="Densidad")
        ax.legend(fontsize=8)

    # Ocultar ejes vacíos
    for j in range(len(available_num), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribución de Features Numéricas por Clase de Churn", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "feature_distributions.png", dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: feature_distributions.png")

    # ── 4. Mapa de Correlaciones ──────────────────────────────────────────────
    corr_data = feat_matrix[available_num + ["churn_label"]].corr()
    fig, ax   = plt.subplots(figsize=(12, 10))
    mask      = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(
        corr_data, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Mapa de Correlaciones (Features + Target)", fontsize=13)
    plt.tight_layout()
    plt.savefig(Path(FIGURES_DIR) / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: correlation_heatmap.png")

    # ── 5. Estadísticas Descriptivas ──────────────────────────────────────────
    desc_stats = feat_matrix[available_num].describe().round(3)

    # ── 6. Correlación de Features con Target ─────────────────────────────────
    target_corr = (
        feat_matrix[available_num + ["churn_label"]]
        .corr()["churn_label"]
        .drop("churn_label")
        .sort_values(ascending=False)
    )

    report = {
        "n_customers":    len(feat_matrix),
        "n_features":     feat_matrix.shape[1] - 1,
        "churn_rate":     float(churn_rate),
        "n_churn":        int(churn_counts.get(1, 0)),
        "n_no_churn":     int(churn_counts.get(0, 0)),
        "null_report":    null_report.to_dict(orient="records"),
        "desc_stats":     desc_stats.to_dict(),
        "target_corr":    target_corr.to_dict(),
        "figures": {
            "class_dist":    str(Path(FIGURES_DIR) / "class_distribution.png"),
            "distributions": str(Path(FIGURES_DIR) / "feature_distributions.png"),
            "correlations":  str(Path(FIGURES_DIR) / "correlation_heatmap.png"),
        },
    }

    logger.info(
        f"✅ EDA Report completado:\n"
        f"   Clientes: {report['n_customers']:,}\n"
        f"   Churn rate: {report['churn_rate']:.1%}\n"
        f"   Features con nulos: {len(null_report)}"
    )
    return report


if __name__ == "__main__":
    report = generate_eda_report()
    print(f"\nTop 5 correlaciones con Churn:")
    for feat, corr in list(report["target_corr"].items())[:5]:
        print(f"  {feat}: {corr:.3f}")
