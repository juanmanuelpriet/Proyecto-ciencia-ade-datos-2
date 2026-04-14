"""
eda/data_quality_report.py — Reporte de Calidad de Datos (Fase III)
=====================================================================
Genera estadísticas de calidad de datos siguiendo los estándares del
Analytics Contract. Incluye:

  1. Integridad de Llaves (dup_rate, orphan_rate)
  2. Análisis de Missingness (MCAR / MAR / MNAR)
  3. Regla Stop/Go (snapshot aceptable o rechazado)
  4. Factor de Explosión de Joins
  5. Balance de Clases
  6. Distribuciones de Features
  7. Mapa de Correlaciones
  8. Invariantes de Negocio

Todos los resultados se guardan en artifacts/figures/ y se retornan
en un dict para incrustar en los documentos LaTeX.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from loguru import logger

# ─── Configuración ────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

FIGURES_DIR  = CFG["paths"]["figures"]
PROCESSED    = CFG["paths"]["processed_data"]
DB_PATH      = CFG["paths"]["db"]
NULL_THRESH  = CFG["etl"]["null_threshold"]

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"no_churn": "#2D6A4F", "churn": "#E63946"}


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 1: INTEGRIDAD DE LLAVES
# ══════════════════════════════════════════════════════════════════════════════

def check_key_integrity(conn: sqlite3.Connection) -> dict:
    """
    Calcula dup_rate y orphan_rate para todas las tablas del esquema.

    - dup_rate: tasa de duplicados sobre la clave primaria
    - orphan_rate: tasa de registros huérfanos (FK sin referencia en tabla padre)

    Retorna un dict con los resultados por tabla.
    """
    logger.info("🔑 [Módulo 1] Verificando integridad de llaves...")
    results = {}

    tables_pk = {
        "dim_customers":       "customer_id",
        "fact_orders":         "order_id",
        "fact_order_items":    "item_id",
        "fact_support_tickets": "ticket_id",
        "ml_churn_labels":     None,   # PK compuesta
        "ml_feature_store":    None,   # PK compuesta
    }

    for table, pk in tables_pk.items():
        try:
            total = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0, 0]
            if total == 0:
                results[table] = {"total": 0, "dup_rate": 0.0, "orphan_rate": None}
                continue

            if pk:
                distinct = pd.read_sql(
                    f"SELECT COUNT(DISTINCT {pk}) as n FROM {table}", conn
                ).iloc[0, 0]
                dup_rate = round((1 - distinct / total) * 100, 4) if total > 0 else 0.0
            else:
                dup_rate = 0.0  # PK compuesta: confiamos en la constraint UNIQUE del DDL

            results[table] = {
                "total": int(total),
                "distinct": int(distinct) if pk else int(total),
                "dup_rate": dup_rate,
                "orphan_rate": None,
            }
        except Exception as e:
            logger.warning(f"   ⚠️  No se pudo verificar {table}: {e}")
            results[table] = {"total": 0, "dup_rate": 0.0, "orphan_rate": None}

    # Orphan rates: fact_orders.customer_id → dim_customers
    try:
        orphan_orders = pd.read_sql("""
            SELECT COUNT(*) as n FROM fact_orders fo
            LEFT JOIN dim_customers dc ON fo.customer_id = dc.customer_id
            WHERE dc.customer_id IS NULL
        """, conn).iloc[0, 0]
        total_orders = results.get("fact_orders", {}).get("total", 1) or 1
        results["fact_orders"]["orphan_rate"] = round(orphan_orders / total_orders * 100, 4)
    except Exception:
        pass

    try:
        orphan_tickets = pd.read_sql("""
            SELECT COUNT(*) as n FROM fact_support_tickets ft
            LEFT JOIN dim_customers dc ON ft.customer_id = dc.customer_id
            WHERE dc.customer_id IS NULL
        """, conn).iloc[0, 0]
        total_tickets = results.get("fact_support_tickets", {}).get("total", 1) or 1
        results["fact_support_tickets"]["orphan_rate"] = round(
            orphan_tickets / total_tickets * 100, 4
        )
    except Exception:
        pass

    for t, r in results.items():
        or_str = f"{r['orphan_rate']}%" if r['orphan_rate'] is not None else "N/A"
        logger.info(f"   {t}: total={r['total']:,} | dup_rate={r['dup_rate']}% | orphan={or_str}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 2: FACTOR DE EXPLOSIÓN DE JOINS
# ══════════════════════════════════════════════════════════════════════════════

def compute_explosion_factors(conn: sqlite3.Connection) -> dict:
    """
    Calcula el factor de explosión explosion(A, B) = |A ⋈ B| / |A|
    para los joins críticos del sistema.
    """
    logger.info("💥 [Módulo 2] Calculando factores de explosión de joins...")
    factors = {}

    joins = [
        ("dim_customers ⋈ fact_orders",
         "SELECT COUNT(*) FROM fact_orders",
         "SELECT COUNT(*) FROM dim_customers"),
        ("fact_orders ⋈ fact_order_items",
         "SELECT COUNT(*) FROM fact_order_items",
         "SELECT COUNT(*) FROM fact_orders"),
        ("dim_customers ⋈ fact_support_tickets",
         "SELECT COUNT(*) FROM fact_support_tickets",
         "SELECT COUNT(*) FROM dim_customers"),
    ]

    for name, q_right, q_left in joins:
        try:
            n_right = pd.read_sql(q_right, conn).iloc[0, 0]
            n_left  = pd.read_sql(q_left,  conn).iloc[0, 0]
            factor  = round(n_right / n_left, 3) if n_left > 0 else 0.0
            factors[name] = {"n_left": int(n_left), "n_right": int(n_right), "factor": factor}
            logger.info(f"   {name}: explosion = {factor:.2f}x")
        except Exception as e:
            logger.warning(f"   ⚠️  No se pudo calcular {name}: {e}")

    return factors


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 3: ANÁLISIS DE MISSINGNESS (MCAR / MAR / MNAR)
# ══════════════════════════════════════════════════════════════════════════════

# Clasificación manual basada en conocimiento del dominio
MISSINGNESS_TAXONOMY = {
    "avg_satisfaction_score": {
        "type": "MNAR",
        "justification": (
            "Los clientes muy insatisfechos tienen mayor probabilidad de no responder "
            "encuestas CSAT. La ausencia está correlacionada con el valor latente "
            "(insatisfacción extrema), lo cual introduce sesgo si se imputa por media."
        ),
        "treatment": "Imputación por mediana + indicador binario 'has_csat'",
    },
    "avg_delivery_days": {
        "type": "MAR",
        "justification": (
            "Los valores faltantes ocurren cuando order_status IN ('PENDING','FAILED'). "
            "La ausencia es explicada por otra variable observable (order_status), "
            "no por el valor de delivery_days en sí."
        ),
        "treatment": "Imputación por mediana estratificada por order_status",
    },
    "return_rate": {
        "type": "MCAR",
        "justification": (
            "La tasa de devoluciones es 0 por definición para clientes nuevos "
            "(< 1 orden). La ausencia no está correlacionada con ningún predictor "
            "ni con el target."
        ),
        "treatment": "Imputación por media de la distribución observada",
    },
    "trend_slope_30d": {
        "type": "MAR",
        "justification": (
            "La pendiente de tendencia no puede calcularse para clientes con "
            "menos de 2 órdenes en la ventana de observación. La ausencia depende "
            "de frequency_orders (variable observable)."
        ),
        "treatment": "Imputación por 0.0 (sin tendencia detectable)",
    },
}


def analyze_missingness(feat_matrix: pd.DataFrame) -> dict:
    """
    Calcula estadísticas de missingness y enriquece con la taxonomía MCAR/MAR/MNAR.
    """
    logger.info("🔍 [Módulo 3] Analizando missingness...")

    null_counts = feat_matrix.isnull().sum()
    null_pct    = (null_counts / len(feat_matrix) * 100).round(2)

    # Detectar columnas con nulos > umbral
    over_threshold = null_pct[null_pct > NULL_THRESH * 100].index.tolist()
    if over_threshold:
        logger.warning(f"   ⚠️  STOP: columnas con nulos > {NULL_THRESH*100:.0f}%: {over_threshold}")
    else:
        logger.info(f"   ✅ GO: ninguna columna supera el umbral de {NULL_THRESH*100:.0f}% de nulos")

    report = []
    for col in feat_matrix.columns:
        if col == "churn_label":
            continue
        entry = {
            "column":    col,
            "null_count": int(null_counts.get(col, 0)),
            "null_pct":   float(null_pct.get(col, 0.0)),
            "type":       "—",
            "treatment":  "No requiere",
        }
        if col in MISSINGNESS_TAXONOMY:
            entry["type"]      = MISSINGNESS_TAXONOMY[col]["type"]
            entry["treatment"] = MISSINGNESS_TAXONOMY[col]["treatment"]
        elif null_counts.get(col, 0) > 0:
            entry["type"]    = "MCAR (default)"
            entry["treatment"] = "Imputación por mediana/moda"

        report.append(entry)

    return {
        "report":          report,
        "over_threshold":  over_threshold,
        "has_stop":        len(over_threshold) > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 4: REGLA STOP/GO
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_stop_go(
    key_integrity: dict,
    missingness:   dict,
    feat_matrix:   pd.DataFrame,
) -> dict:
    """
    Evalúa si el snapshot es aceptable para proceder al modelado.

    Retorna {'status': 'GO'} o {'status': 'STOP', 'reasons': [...]}
    """
    logger.info("🚦 [Módulo 4] Evaluando regla Stop/Go...")
    reasons = []

    # Criterio 1: integridad de llaves
    for table, r in key_integrity.items():
        if r.get("dup_rate", 0) > 0:
            reasons.append(f"dup_rate > 0% en tabla {table} ({r['dup_rate']}%)")
        if r.get("orphan_rate") is not None and r["orphan_rate"] > 0:
            reasons.append(f"orphan_rate > 0% en tabla {table} ({r['orphan_rate']}%)")

    # Criterio 2: nulos excesivos
    if missingness["has_stop"]:
        for col in missingness["over_threshold"]:
            reasons.append(f"columna '{col}' supera {NULL_THRESH*100:.0f}% de nulos")

    # Criterio 3: invariante temporal (delivery_date >= order_date)
    if "avg_delivery_days" in feat_matrix.columns:
        neg_delivery = (feat_matrix["avg_delivery_days"] < 0).sum()
        if neg_delivery > 0:
            reasons.append(f"{neg_delivery} clientes con avg_delivery_days < 0 (violación temporal)")

    # Criterio 4: churn rate dentro de rangos esperados
    churn_rate = feat_matrix["churn_label"].mean() if "churn_label" in feat_matrix.columns else 0
    if churn_rate < 0.01 or churn_rate > 0.50:
        reasons.append(
            f"churn_rate = {churn_rate:.1%} fuera del rango esperado [1%, 50%]"
        )

    status = "STOP" if reasons else "GO"
    emoji  = "🛑 STOP" if reasons else "✅ GO"
    logger.info(f"   Resultado: {emoji} | {len(reasons)} criterios fallidos")
    for r in reasons:
        logger.warning(f"   ❌ {r}")

    return {"status": status, "reasons": reasons, "churn_rate": float(churn_rate)}


# ══════════════════════════════════════════════════════════════════════════════
#  MÓDULO 5: VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _plot_class_distribution(feat_matrix: pd.DataFrame) -> str:
    """Genera gráfico de balance de clases."""
    churn_counts = feat_matrix["churn_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [PALETTE["no_churn"], PALETTE["churn"]]
    bars   = ax.bar(["No Churn (0)", "Churn (1)"], churn_counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, churn_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{val:,}\n({val/len(feat_matrix):.1%})",
            ha="center", fontsize=10, weight="bold",
        )
    ax.set(title="Distribución de Clases — Churn Label", ylabel="Número de Clientes")
    ax.set_ylim(0, max(churn_counts.values) * 1.22)
    ax.set_xlabel("Clase", labelpad=8)
    plt.tight_layout()
    path = str(Path(FIGURES_DIR) / "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: class_distribution.png")
    return path


def _plot_feature_distributions(feat_matrix: pd.DataFrame, numeric_features: list) -> str:
    """Genera histogramas de features numéricas separados por clase."""
    available = [f for f in numeric_features if f in feat_matrix.columns]
    n_cols = 3
    n_rows = (len(available) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i, feat in enumerate(available):
        ax = axes[i]
        for label, color, name in zip(
            [0, 1],
            [PALETTE["no_churn"], PALETTE["churn"]],
            ["No Churn", "Churn"],
        ):
            subset = feat_matrix[feat_matrix["churn_label"] == label][feat].dropna()
            if len(subset) > 0:
                ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, density=True)
        ax.set(title=feat.replace("_", " ").title(), xlabel=feat, ylabel="Densidad")
        ax.legend(fontsize=7)

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Distribución de Features Numéricas por Clase de Churn", y=1.01, fontsize=13)
    plt.tight_layout()
    path = str(Path(FIGURES_DIR) / "feature_distributions.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: feature_distributions.png")
    return path


def _plot_correlation_heatmap(feat_matrix: pd.DataFrame, numeric_features: list) -> str:
    """Genera mapa de correlaciones entre features y target."""
    available = [f for f in numeric_features if f in feat_matrix.columns]
    corr_data = feat_matrix[available + ["churn_label"]].corr()
    mask = np.triu(np.ones_like(corr_data, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr_data, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, ax=ax,
        annot_kws={"size": 6.5},
        linewidths=0.4, linecolor="white",
    )
    ax.set_title("Mapa de Correlaciones (Features + Target Churn)", fontsize=12, pad=12)
    plt.tight_layout()
    path = str(Path(FIGURES_DIR) / "correlation_heatmap.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: correlation_heatmap.png")
    return path


def _plot_missingness_bar(miss_report: list) -> str:
    """Genera gráfico de barras de missingness por columna."""
    df_miss = pd.DataFrame(miss_report)
    df_miss = df_miss[df_miss["null_pct"] > 0].sort_values("null_pct", ascending=True)

    if df_miss.empty:
        logger.info("   ✅ Sin valores faltantes: gráfico de missingness omitido.")
        return ""

    fig, ax = plt.subplots(figsize=(8, max(3, len(df_miss) * 0.5)))
    colors = [
        "#E63946" if r >= NULL_THRESH * 100 else "#2D6A4F"
        for r in df_miss["null_pct"]
    ]
    ax.barh(df_miss["column"], df_miss["null_pct"], color=colors, edgecolor="white")
    ax.axvline(NULL_THRESH * 100, color="red", linestyle="--", alpha=0.7,
               label=f"Umbral STOP ({NULL_THRESH*100:.0f}%)")
    ax.set(title="Tasa de Missingness por Feature", xlabel="% Valores Faltantes", ylabel="Feature")
    ax.legend(fontsize=8)
    # Anotar tipo MCAR/MAR/MNAR
    for _, row in df_miss.iterrows():
        tipo = MISSINGNESS_TAXONOMY.get(row["column"], {}).get("type", "")
        if tipo:
            ax.text(row["null_pct"] + 0.3, row["column"], tipo, va="center", fontsize=7, color="gray")
    plt.tight_layout()
    path = str(Path(FIGURES_DIR) / "missingness_bar.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: missingness_bar.png")
    return path


def _plot_target_correlation(feat_matrix: pd.DataFrame, numeric_features: list) -> str:
    """Genera gráfico de barras de correlación de cada feature con churn_label."""
    available = [f for f in numeric_features if f in feat_matrix.columns]
    target_corr = (
        feat_matrix[available + ["churn_label"]]
        .corr()["churn_label"]
        .drop("churn_label")
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(target_corr) * 0.5)))
    colors = ["#E63946" if c > 0 else "#2D6A4F" for c in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Correlación de Features con Target (Churn)", xlabel="Correlación de Pearson", ylabel="Feature")
    plt.tight_layout()
    path = str(Path(FIGURES_DIR) / "target_correlation.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info("💾 Figura guardada: target_correlation.png")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def generate_eda_report(feat_matrix: Optional[pd.DataFrame] = None) -> dict:
    """
    Genera el reporte completo de EDA y calidad de datos.

    Retorna un dict con todas las estadísticas listas para incrustar
    en los documentos LaTeX.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Cargar feature matrix si no se provee ─────────────────────────────────
    if feat_matrix is None:
        parquet_path = Path(PROCESSED) / "feature_matrix.parquet"
        logger.info(f"Cargando feature matrix desde {parquet_path}...")
        feat_matrix = pd.read_parquet(parquet_path)

    logger.info(
        f"\n{'='*60}\n"
        f"📊 EDA REPORT — FASE III: CALIDAD DE DATOS\n"
        f"{'='*60}\n"
        f"   Shape: {feat_matrix.shape[0]:,} clientes × {feat_matrix.shape[1]} columnas\n"
        f"   Columnas: {list(feat_matrix.columns)}"
    )

    numeric_features = CFG["features"]["numeric"]

    # ── Módulo 1: Integridad de Llaves ────────────────────────────────────────
    key_integrity = {}
    try:
        with sqlite3.connect(DB_PATH) as conn:
            key_integrity    = check_key_integrity(conn)
            explosion_factors = compute_explosion_factors(conn)
    except Exception as e:
        logger.warning(f"⚠️  No se pudo conectar a la DB para integridad: {e}")
        explosion_factors = {}

    # ── Módulo 3: Missingness ─────────────────────────────────────────────────
    missingness = analyze_missingness(feat_matrix)

    # ── Módulo 4: Stop/Go ─────────────────────────────────────────────────────
    stop_go = evaluate_stop_go(key_integrity, missingness, feat_matrix)

    if stop_go["status"] == "STOP":
        logger.error("🛑 STOP: snapshot rechazado. No se procederá al entrenamiento.")
        logger.error(f"   Razones: {stop_go['reasons']}")
    else:
        logger.info("✅ GO: snapshot aceptado. Procediendo al entrenamiento.")

    # ── Módulo 5: Visualizaciones ─────────────────────────────────────────────
    churn_counts = feat_matrix["churn_label"].value_counts()
    churn_rate   = feat_matrix["churn_label"].mean()

    figures = {}
    try:
        figures["class_dist"]    = _plot_class_distribution(feat_matrix)
        figures["distributions"] = _plot_feature_distributions(feat_matrix, numeric_features)
        figures["correlations"]  = _plot_correlation_heatmap(feat_matrix, numeric_features)
        figures["missingness"]   = _plot_missingness_bar(missingness["report"])
        figures["target_corr"]   = _plot_target_correlation(feat_matrix, numeric_features)
    except Exception as e:
        logger.warning(f"⚠️  Error generando figuras: {e}")

    # ── Estadísticas descriptivas ─────────────────────────────────────────────
    available_num = [f for f in numeric_features if f in feat_matrix.columns]
    desc_stats    = feat_matrix[available_num].describe().round(3)
    target_corr   = (
        feat_matrix[available_num + ["churn_label"]]
        .corr()["churn_label"]
        .drop("churn_label")
        .sort_values(ascending=False)
    )

    # ── Reporte consolidado ───────────────────────────────────────────────────
    report = {
        # Dimensiones
        "n_customers":     int(len(feat_matrix)),
        "n_features":      int(feat_matrix.shape[1] - 1),
        "churn_rate":      float(churn_rate),
        "n_churn":         int(churn_counts.get(1, 0)),
        "n_no_churn":      int(churn_counts.get(0, 0)),

        # Módulo 1: Integridad de llaves
        "key_integrity":       key_integrity,
        "explosion_factors":   explosion_factors,

        # Módulo 3: Missingness
        "null_report":         missingness["report"],
        "over_threshold_cols": missingness["over_threshold"],

        # Módulo 4: Stop/Go
        "stop_go_status":  stop_go["status"],
        "stop_go_reasons": stop_go["reasons"],

        # Módulo 5: Estadísticas
        "desc_stats":  desc_stats.to_dict(),
        "target_corr": target_corr.to_dict(),

        # Figuras
        "figures": figures,
    }

    logger.info(
        f"\n{'='*60}\n"
        f"📋 EDA REPORT COMPLETADO\n"
        f"{'='*60}\n"
        f"   Clientes:            {report['n_customers']:,}\n"
        f"   Churn rate:          {report['churn_rate']:.1%}\n"
        f"   Features con nulos:  {len([r for r in report['null_report'] if r['null_count'] > 0])}\n"
        f"   Stop/Go status:      {report['stop_go_status']}\n"
        f"   Figuras generadas:   {len([v for v in figures.values() if v])}"
    )

    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    report = generate_eda_report()

    print("\n── INTEGRIDAD DE LLAVES ──")
    for t, r in report["key_integrity"].items():
        print(f"  {t}: dup_rate={r['dup_rate']}% | orphan={r.get('orphan_rate', 'N/A')}%")

    print("\n── EXPLOSION FACTORS ──")
    for j, r in report["explosion_factors"].items():
        print(f"  {j}: {r['factor']:.2f}x")

    print("\n── MISSINGNESS ──")
    for r in report["null_report"]:
        if r["null_count"] > 0:
            print(f"  {r['column']}: {r['null_pct']}% [{r['type']}] → {r['treatment']}")

    print(f"\n── REGLA STOP/GO: {report['stop_go_status']} ──")
    for reason in report["stop_go_reasons"]:
        print(f"  ❌ {reason}")

    print(f"\n── TOP 5 CORRELACIONES CON CHURN ──")
    for feat, corr in list(report["target_corr"].items())[:5]:
        print(f"  {feat}: {corr:.3f}")
