"""
etl/feature_engineer.py — Feature Engineering Point-in-Time Correcto
======================================================================
Calcula todas las features USANDO SOLO información disponible antes de
cutoff_date. Jamás usa información del futuro como input.

FEATURES IMPLEMENTADAS:
1. RFM clásico (Recency, Frequency, Monetary)
2. Trend (señal de deterioro de gasto)
3. Diversity (diversidad de categorías)
4. Support Load (fricción operacional)
5. Logistics (calidad de entrega)
6. Tenure (antigüedad del cliente)

DISEÑO: Cada función recibe DataFrames completos y retorna una serie/columna.
Esto permite una composición limpia en el pipeline.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.stats import linregress


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

CUTOFF_DATE = date.fromisoformat(CFG["temporal"]["cutoff_date"])
OBS_WINDOW  = CFG["temporal"]["observation_window_days"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _obs_orders(orders: pd.DataFrame, cutoff: date, window_days: int) -> pd.DataFrame:
    """Filtra órdenes dentro de la ventana de observación."""
    obs_start = cutoff - timedelta(days=window_days)
    mask = (
        (orders["order_date"].dt.date >= obs_start) &
        (orders["order_date"].dt.date <= cutoff) &
        (orders["order_status"] == "COMPLETED")  # Solo órdenes exitosas
    )
    return orders[mask].copy()


# ─── Feature: Recency ─────────────────────────────────────────────────────────

def compute_recency(
    orders: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    window_days: int = OBS_WINDOW,
) -> pd.Series:
    """
    Días desde la última compra COMPLETADA hasta cutoff_date.
    Mayor valor → más tiempo sin comprar → mayor riesgo de churn.
    """
    obs = _obs_orders(orders, cutoff, window_days)
    last_buy = obs.groupby("customer_id")["order_date"].max()
    recency  = (pd.Timestamp(cutoff) - last_buy).dt.days
    return recency.rename("recency_days")


# ─── Feature: Frequency ───────────────────────────────────────────────────────

def compute_frequency(
    orders: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    window_days: int = OBS_WINDOW,
) -> pd.Series:
    """
    Número de órdenes completadas en la ventana de observación.
    """
    obs = _obs_orders(orders, cutoff, window_days)
    freq = obs.groupby("customer_id")["order_id"].count()
    return freq.rename("frequency_orders")


# ─── Feature: Monetary ────────────────────────────────────────────────────────

def compute_monetary(
    orders: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    window_days: int = OBS_WINDOW,
) -> pd.DataFrame:
    """
    Gasto total y promedio por orden en la ventana.
    Se retorna un DataFrame porque son 2 features.

    ADICIÓN (INTENSITY): Se agrega 'spending_intensity' (gasto per cápita diario
    en la ventana de observación).
    """
    obs   = _obs_orders(orders, cutoff, window_days)
    agg   = obs.groupby("customer_id")["total_amount"].agg(
        monetary_total="sum",
        monetary_avg_order="mean",
    ).copy()
    
    # Intensidad: Gasto por día en la ventana
    agg["spending_intensity"] = agg["monetary_total"] / window_days
    
    return agg


# ─── Feature: Trend (señal de deterioro) ──────────────────────────────────────

def compute_trend_slope(
    orders: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    short_window_days: int = 30,
    long_window_days: int  = OBS_WINDOW,
) -> pd.Series:
    """
    Calcula la pendiente de la regresión lineal del gasto por semana
    en los últimos `short_window_days` días dentro de la ventana de observación.

    INTERPRETACIÓN:
    - Pendiente negativa = gasto bajando → señal de churn inminente
    - Pendiente positiva = cliente mejor que antes → baja probabilidad de churn

    DECISIÓN: usamos regresión lineal simple porque es interpretable y
    suficientemente robusta para séries de tiempo cortas (~4 puntos).
    """
    obs = _obs_orders(orders, cutoff, long_window_days)
    # Agregar gasto por semana
    obs = obs.copy()
    obs["week"] = obs["order_date"].dt.isocalendar().week.astype(int)
    obs["year"] = obs["order_date"].dt.year.astype(int)
    obs["year_week"] = obs["year"] * 100 + obs["week"]

    weekly = obs.groupby(["customer_id", "year_week"])["total_amount"].sum().reset_index()

    def _slope(group: pd.Series) -> float:
        if len(group) < 2:
            return 0.0  # Sin suficientes puntos → sin tendencia
        x = np.arange(len(group))
        try:
            slope, _, _, _, _ = linregress(x, group["total_amount"].values)
            return float(slope)
        except Exception:
            return 0.0

    slopes = weekly.groupby("customer_id").apply(_slope)
    return slopes.rename("trend_slope_30d")


# ─── Feature: Category Diversity ──────────────────────────────────────────────

def compute_category_diversity(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    window_days: int = OBS_WINDOW,
) -> pd.Series:
    """
    Número de categorías de productos distintas compradas en la ventana.
    Clientes con baja diversidad son más vulnerables (ej: solo compran Electronics
    y si ese producto falla, se van).
    """
    obs = _obs_orders(orders, cutoff, window_days)
    merged = obs.merge(order_items[["order_id", "category"]], on="order_id", how="left")
    diversity = merged.groupby("customer_id")["category"].nunique()
    return diversity.rename("category_diversity")


# ─── Feature: Support Load ────────────────────────────────────────────────────

def compute_support_features(
    tickets: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    window_days: int = OBS_WINDOW,
) -> pd.DataFrame:
    """
    Número de tickets y NPS promedio en la ventana.
    Tickets de soporte tienen alta correlación con churn (señal de fricción).
    """
    obs_start = cutoff - timedelta(days=window_days)
    mask = (
        (tickets["created_date"].dt.date >= obs_start) &
        (tickets["created_date"].dt.date <= cutoff)
    )
    obs = tickets[mask].copy()

    agg = obs.groupby("customer_id").agg(
        support_tickets_90d=("ticket_id", "count"),
        avg_satisfaction_score=("satisfaction_score", "mean"),
    )
    return agg


# ─── Feature: Logistics ───────────────────────────────────────────────────────

def compute_logistics_features(
    orders: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
    window_days: int = OBS_WINDOW,
) -> pd.DataFrame:
    """
    Días promedio de entrega y tasa de devoluciones.
    Mala logística → insatisfacción → churn.
    """
    obs_all = orders[
        (orders["order_date"].dt.date >= (cutoff - timedelta(days=window_days))) &
        (orders["order_date"].dt.date <= cutoff)
    ].copy()

    # Días de entrega (solo para completadas)
    completed = obs_all[obs_all["order_status"] == "COMPLETED"].copy()
    completed["delivery_days"] = (
        completed["delivery_date"] - completed["order_date"]
    ).dt.days

    avg_del = completed.groupby("customer_id")["delivery_days"].mean().rename("avg_delivery_days")

    # Tasa de devoluciones
    total_orders  = obs_all.groupby("customer_id")["order_id"].count()
    returned      = obs_all[obs_all["order_status"] == "RETURNED"].groupby("customer_id")["order_id"].count()
    return_rate   = (returned / total_orders).fillna(0).rename("return_rate")

    return pd.concat([avg_del, return_rate], axis=1)


# ─── Feature: Tenure ──────────────────────────────────────────────────────────

def compute_tenure(
    customers: pd.DataFrame,
    cutoff: date = CUTOFF_DATE,
) -> pd.Series:
    """
    Días desde el registro del cliente hasta cutoff.
    Clientes nuevos tienen más probabilidad de churn (curva de aprendizaje del producto).
    """
    customers = customers.set_index("customer_id")
    tenure = (pd.Timestamp(cutoff) - customers["registration_date"]).dt.days
    return tenure.rename("days_since_registration")


# ─── Ensamblador Principal ────────────────────────────────────────────────────

def build_feature_matrix(
    customers:    pd.DataFrame,
    orders:       pd.DataFrame,
    order_items:  pd.DataFrame,
    tickets:      pd.DataFrame,
    labels:       pd.DataFrame,
    cutoff:       date = CUTOFF_DATE,
    window_days:  int  = OBS_WINDOW,
) -> pd.DataFrame:
    """
    Construye la matriz de features completa uniendo todas las señales.
    Retorna un DataFrame indexado por customer_id, con el target 'churn_label'.

    ANTI-LEAKAGE CHECKLIST:
    ✅ Todas las features usan datos <= cutoff_date
    ✅ Las etiquetas de churn usan datos > cutoff_date (ventana de predicción)
    ✅ El split train/test se aplica DESPUÉS de esta función
    ✅ SMOTE se aplica solo en el train set
    """
    logger.info("🔧 Calculando features...")

    # Calcular cada señal
    recency    = compute_recency(orders, cutoff, window_days)
    frequency  = compute_frequency(orders, cutoff, window_days)
    monetary   = compute_monetary(orders, cutoff, window_days)
    trend      = compute_trend_slope(orders, cutoff)
    diversity  = compute_category_diversity(orders, order_items, cutoff, window_days)
    support    = compute_support_features(tickets, cutoff, window_days)
    logistics  = compute_logistics_features(orders, cutoff, window_days)
    tenure     = compute_tenure(customers, cutoff)

    # Unir todo en una matriz (outer join → nulos para clientes sin actividad)
    feat_matrix = pd.concat(
        [recency, frequency, monetary, trend, diversity, support, logistics, tenure],
        axis=1,
    )

    # Join con clientes para tener variables categóricas
    cust_meta = customers.set_index("customer_id")[
        ["customer_segment", "preferred_category"]
    ]
    feat_matrix = feat_matrix.join(cust_meta, how="left")

    # Join con labels
    labels_idx = labels.set_index("customer_id")["churn_label"]
    feat_matrix = feat_matrix.join(labels_idx, how="inner")  # Inner: solo clientes etiquetados

    # Imputar nulos residuales
    numeric_cols = feat_matrix.select_dtypes(include=[np.number]).columns
    feat_matrix[numeric_cols] = feat_matrix[numeric_cols].fillna(
        feat_matrix[numeric_cols].median()
    )
    feat_matrix["customer_segment"].fillna("STANDARD", inplace=True)
    feat_matrix["preferred_category"].fillna("Unknown", inplace=True)
    feat_matrix["support_tickets_90d"].fillna(0, inplace=True)

    logger.info(
        f"✅ Feature matrix: {feat_matrix.shape[0]:,} clientes × {feat_matrix.shape[1]} features | "
        f"Churn rate: {feat_matrix['churn_label'].mean():.1%}"
    )
    return feat_matrix
