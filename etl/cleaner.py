"""
etl/cleaner.py — Limpieza, Validación de Invariantes y Calidad de Datos
========================================================================
Esta capa implementa las siguientes responsabilidades:
1. Validación de invariantes de negocio (reglas que NUNCA deben violarse)
2. Tratamiento de nulos (estrategia diferenciada por tipo de variable)
3. Capping de outliers (IQR robusto, no elimina registros)
4. Normalización de tipos de datos

DECISIÓN sobre nulos: Se prefiere imputación sobre eliminación porque
eliminar filas en un dataset de ML introduce sesgo de selección.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

NULL_THRESHOLD = CFG["etl"]["null_threshold"]
IQR_FACTOR     = CFG["etl"]["iqr_factor"]


# ─── Resultado de validación de invariantes ───────────────────────────────────
@dataclass
class InvariantResult:
    name:    str
    passed:  bool
    n_violations: int = 0
    details: str = ""


@dataclass
class CleaningReport:
    """Reporte de limpieza para auditoría y LaTeX."""
    dropped_columns:      list[str] = field(default_factory=list)
    imputed_columns:      dict[str, Any] = field(default_factory=dict)
    outlier_capped:       dict[str, int] = field(default_factory=dict)
    invariant_results:    list[InvariantResult] = field(default_factory=list)
    rows_before:          int = 0
    rows_after:           int = 0

    def to_dict(self) -> dict:
        return {
            "dropped_columns":  self.dropped_columns,
            "imputed_columns":  self.imputed_columns,
            "outlier_capped":   self.outlier_capped,
            "invariant_results": [
                {"name": r.name, "passed": r.passed, "violations": r.n_violations}
                for r in self.invariant_results
            ],
            "rows_before": self.rows_before,
            "rows_after":  self.rows_after,
        }


# ─── Validadores de Invariantes ───────────────────────────────────────────────

def _check_delivery_after_order(orders: pd.DataFrame) -> InvariantResult:
    """
    INVARIANTE: La fecha de entrega debe ser >= fecha de orden.
    Violaciones pueden indicar errores de ingesta o de zona horaria.
    """
    mask = (
        orders["delivery_date"].notna() &
        (pd.to_datetime(orders["delivery_date"]) < pd.to_datetime(orders["order_date"]))
    )
    n_viol = int(mask.sum())
    if n_viol > 0:
        logger.warning(f"⚠️  INVARIANTE delivery_after_order: {n_viol} violaciones. Limpiando...")
        orders.loc[mask, "delivery_date"] = None  # Convertir a NULL: dato incierto
    return InvariantResult(
        name="delivery_after_order",
        passed=(n_viol == 0),
        n_violations=n_viol,
    )


def _check_positive_amounts(orders: pd.DataFrame) -> InvariantResult:
    """
    INVARIANTE: El monto total de la orden debe ser >= 0.
    Montos negativos son un error de datos (no devoluciones legítimas, esas son RETURNED).
    """
    mask  = orders["total_amount"] < 0
    n_viol = int(mask.sum())
    if n_viol > 0:
        logger.warning(f"⚠️  INVARIANTE positive_amounts: {n_viol} violaciones. Imputando mediana...")
        median_val = orders.loc[~mask, "total_amount"].median()
        orders.loc[mask, "total_amount"] = median_val
    return InvariantResult(
        name="positive_amounts",
        passed=(n_viol == 0),
        n_violations=n_viol,
    )


def _check_satisfaction_range(tickets: pd.DataFrame) -> InvariantResult:
    """
    INVARIANTE: satisfaction_score debe estar entre 1 y 5 (o ser NULL).
    """
    mask  = tickets["satisfaction_score"].notna() & (
        (tickets["satisfaction_score"] < 1) | (tickets["satisfaction_score"] > 5)
    )
    n_viol = int(mask.sum())
    if n_viol > 0:
        logger.warning(f"⚠️  INVARIANTE satisfaction_range: {n_viol} violaciones. Convirtiendo a NULL...")
        tickets.loc[mask, "satisfaction_score"] = np.nan
    return InvariantResult(
        name="satisfaction_range",
        passed=(n_viol == 0),
        n_violations=n_viol,
    )


# ─── Limpieza por DataFrame ───────────────────────────────────────────────────

def clean_orders(orders: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Limpieza del DataFrame de órdenes."""
    report              = CleaningReport()
    report.rows_before  = len(orders)

    # 1. Parsear fechas
    orders["order_date"]    = pd.to_datetime(orders["order_date"], errors="coerce")
    orders["delivery_date"] = pd.to_datetime(orders["delivery_date"], errors="coerce")

    # 2. Validar invariantes
    report.invariant_results.append(_check_delivery_after_order(orders))
    report.invariant_results.append(_check_positive_amounts(orders))

    # 3. Eliminar columnas con demasiados nulos
    null_pct = orders.isnull().mean()
    to_drop  = null_pct[null_pct > NULL_THRESHOLD].index.tolist()
    if to_drop:
        logger.info(f"🗑  Eliminando columnas con >{NULL_THRESHOLD:.0%} nulos: {to_drop}")
        orders.drop(columns=to_drop, inplace=True)
        report.dropped_columns.extend(to_drop)

    # 4. Imputar nulos en columnas categóricas con moda
    cat_cols = ["payment_method", "channel", "order_status"]
    for col in cat_cols:
        if col in orders.columns and orders[col].isnull().any():
            mode_val = orders[col].mode()[0]
            orders[col].fillna(mode_val, inplace=True)
            report.imputed_columns[col] = f"mode={mode_val}"

    # 5. Capping de outliers en total_amount (IQR robusto)
    if "total_amount" in orders.columns:
        q1  = orders["total_amount"].quantile(0.25)
        q3  = orders["total_amount"].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - IQR_FACTOR * iqr
        hi  = q3 + IQR_FACTOR * iqr
        mask_outlier = (orders["total_amount"] < lo) | (orders["total_amount"] > hi)
        n_capped = int(mask_outlier.sum())
        orders["total_amount"] = orders["total_amount"].clip(lower=lo, upper=hi)
        report.outlier_capped["total_amount"] = n_capped

    report.rows_after = len(orders)
    logger.info(f"✅ clean_orders: {report.rows_before:,} → {report.rows_after:,} filas")
    return orders, report


def clean_support_tickets(tickets: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Limpieza del DataFrame de tickets de soporte."""
    report             = CleaningReport()
    report.rows_before = len(tickets)

    tickets["created_date"]  = pd.to_datetime(tickets["created_date"], errors="coerce")
    tickets["resolved_date"] = pd.to_datetime(tickets["resolved_date"], errors="coerce")

    report.invariant_results.append(_check_satisfaction_range(tickets))

    # Imputar satisfaction_score con mediana (no con moda; es ordinal)
    if tickets["satisfaction_score"].isnull().any():
        med = tickets["satisfaction_score"].median()
        tickets["satisfaction_score"].fillna(med, inplace=True)
        report.imputed_columns["satisfaction_score"] = f"median={med:.1f}"

    report.rows_after = len(tickets)
    return tickets, report


def clean_customers(customers: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Limpieza del maestro de clientes."""
    report             = CleaningReport()
    report.rows_before = len(customers)

    customers["registration_date"] = pd.to_datetime(customers["registration_date"], errors="coerce")

    # Eliminar duplicados de customer_id (si existen por error de ingesta)
    n_dup = customers.duplicated(subset=["customer_id"]).sum()
    if n_dup > 0:
        logger.warning(f"⚠️  {n_dup} customer_ids duplicados. Eliminando...")
        customers.drop_duplicates(subset=["customer_id"], keep="last", inplace=True)

    customers["customer_segment"].fillna("STANDARD", inplace=True)
    customers["preferred_category"].fillna("Unknown", inplace=True)
    customers["email_opt_in"] = customers["email_opt_in"].fillna(1).astype(int)

    report.rows_after = len(customers)
    return customers, report
