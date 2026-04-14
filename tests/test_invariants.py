"""
tests/test_invariants.py — Tests de Invariantes de Negocio
=========================================================
Verifica que los datos generados y procesados cumplan con las reglas críticas.
"""

import pytest
import sqlite3
import pandas as pd
import yaml
from pathlib import Path

# Cargar config
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH = CFG["paths"]["db"]

@pytest.fixture
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()

def test_db_exists():
    """Verifica que el archivo de base de datos se haya creado."""
    assert Path(DB_PATH).exists(), f"La base de datos no se encontró en {DB_PATH}"

def test_invariants_delivery_dates(db_conn):
    """INVARIANTE: delivery_date >= order_date."""
    query = """
    SELECT * FROM fact_orders 
    WHERE delivery_date IS NOT NULL 
    AND delivery_date < order_date
    """
    df = pd.read_sql_query(query, db_conn)
    assert len(df) == 0, f"Se encontraron {len(df)} órdenes con fecha de entrega inválida"

def test_invariants_positive_amounts(db_conn):
    """INVARIANTE: total_amount >= 0."""
    query = "SELECT * FROM fact_orders WHERE total_amount < 0"
    df = pd.read_sql_query(query, db_conn)
    assert len(df) == 0, f"Se encontraron {len(df)} órdenes con montos negativos"

def test_invariants_satisfaction_score(db_conn):
    """INVARIANTE: satisfaction_score entre 1 y 5."""
    query = "SELECT * FROM fact_support_tickets WHERE satisfaction_score < 1 OR satisfaction_score > 5"
    df = pd.read_sql_query(query, db_conn)
    assert len(df) == 0, f"Se encontraron {len(df)} tickets con score fuera de rango"

def test_unique_customers(db_conn):
    """INVARIANTE: customer_id debe ser único."""
    query = "SELECT customer_id, COUNT(*) FROM dim_customers GROUP BY customer_id HAVING COUNT(*) > 1"
    df = pd.read_sql_query(query, db_conn)
    assert len(df) == 0, "Se encontraron customer_id duplicados"

def test_churn_label_binary(db_conn):
    """INVARIANTE: churn_label debe ser 0 o 1."""
    query = "SELECT * FROM ml_churn_labels WHERE churn_label NOT IN (0, 1)"
    df = pd.read_sql_query(query, db_conn)
    assert len(df) == 0, "Se encontraron etiquetas de churn no binarias"
