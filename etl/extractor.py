"""
etl/extractor.py — Capa de Extracción desde SQLite
====================================================
Responsabilidad única: leer datos brutos desde la BD y retornar DataFrames.
No realiza transformaciones; esa es responsabilidad de cleaner.py.

DECISIÓN: Separar extracción de transformación permite:
1. Cambiar la fuente (SQLite → PostgreSQL → BigQuery) sin tocar el resto.
2. Cachear el extractor para agilizar iteraciones de feature engineering.
"""

import sqlite3
from datetime import date
from typing import Optional, Union, Dict

import pandas as pd
import yaml
from loguru import logger


with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = CFG["paths"]["db"]


class RawDataExtractor:
    """
    Extrae datos brutos desde SQLite/PostgreSQL.
    Todos los métodos retornan pd.DataFrame crudos (sin transformación).
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ── Context manager para manejo seguro de conexiones ──────────────────────
    def __enter__(self):
        self._conn = sqlite3.connect(self.db_path)
        return self

    def __exit__(self, *args):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Ejecuta una query y retorna DataFrame. Loguea filas retornadas."""
        df = pd.read_sql_query(sql, self._conn, params=params)
        logger.debug(f"Query retornó {len(df):,} filas.")
        return df

    def get_customers(self) -> pd.DataFrame:
        """Maestro completo de clientes."""
        return self._query("SELECT * FROM dim_customers")

    def get_orders(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Órdenes dentro de un rango de fechas.
        Si no se especifica rango, retorna TODAS las órdenes.
        """
        sql = "SELECT * FROM fact_orders WHERE 1=1"
        params: list = []
        if start_date:
            sql    += " AND order_date >= ?"
            params.append(start_date)
        if end_date:
            sql    += " AND order_date <= ?"
            params.append(end_date)
        sql += " ORDER BY customer_id, order_date"
        return self._query(sql, tuple(params))

    def get_order_items(self) -> pd.DataFrame:
        """
        Line items JOIN con info de producto (categoría).
        Necesitamos la categoría para calcular category_diversity.
        """
        sql = """
            SELECT
                oi.order_id,
                oi.product_id,
                oi.quantity,
                oi.unit_price,
                p.category,
                p.subcategory
            FROM fact_order_items oi
            INNER JOIN dim_products p ON oi.product_id = p.product_id
        """
        return self._query(sql)

    def get_support_tickets(self) -> pd.DataFrame:
        """Tickets de soporte. Señal clave de insatisfacción."""
        return self._query("SELECT * FROM fact_support_tickets")

    def get_churn_labels(self, cutoff_date: Optional[str] = None) -> pd.DataFrame:
        """
        Target labels de churn. Si se especifica cutoff_date, filtra solo
        los labels para esa fecha de corte (permite múltiples cohortes).
        """
        sql    = "SELECT * FROM ml_churn_labels WHERE 1=1"
        params = []
        if cutoff_date:
            sql += " AND cutoff_date = ?"
            params.append(cutoff_date)
        return self._query(sql, tuple(params))

    def get_full_dataset_for_feature_engineering(
        self,
        cutoff_date: str,
        observation_window_days: int = 90,
    ) -> Dict[str, pd.DataFrame]:
        """
        Retorna todos los DataFrames necesarios para feature engineering
        en una sola llamada (evita múltiples conexiones).

        CRÍTICO: Solo extrae datos ANTERIORES al cutoff_date.
        Esto garantiza que no hay data leakage del futuro.
        """
        obs_start = (
            date.fromisoformat(cutoff_date) -
            __import__("datetime").timedelta(days=observation_window_days)
        ).isoformat()

        logger.info(f"🔍 Extrayendo datos | Ventana: {obs_start} → {cutoff_date}")

        return {
            "customers":      self.get_customers(),
            "orders":         self.get_orders(start_date=obs_start, end_date=cutoff_date),
            "order_items":    self.get_order_items(),
            "support_tickets": self.get_support_tickets(),
            "churn_labels":   self.get_churn_labels(cutoff_date=cutoff_date),
        }
