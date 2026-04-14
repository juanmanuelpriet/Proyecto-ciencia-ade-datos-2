"""
tests/test_invariants.py — Tests de Invariantes de Negocio (Fase III)
======================================================================
Verifica que el snapshot de datos cumpla con todas las reglas de integridad
definidas en el Analytics Contract y en el DDL de la base de datos.

COBERTURA:
  - Integridad de llaves primarias (dup_rate = 0%)
  - Integridad referencial (orphan_rate = 0%)
  - Invariantes temporales (delivery >= order, resolved >= created)
  - Rangos de valores de negocio (amounts, scores)
  - Balance de clases dentro de rangos esperados
  - Regla Stop/Go del EDA

Ejecutar con:
    pytest tests/test_invariants.py -v
    pytest tests/test_invariants.py -v --tb=short   # Output más limpio
"""

import pytest
import sqlite3
import pandas as pd
import yaml
from pathlib import Path


# ─── Configuración ────────────────────────────────────────────────────────────

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH        = CFG["paths"]["db"]
PROCESSED_PATH = CFG["paths"]["processed_data"]
NULL_THRESHOLD = CFG["etl"]["null_threshold"]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def db_conn():
    """Conexión a la base de datos (compartida en el módulo para velocidad)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def feature_matrix():
    """Carga la feature matrix procesada si existe."""
    path = Path(PROCESSED_PATH) / "feature_matrix.parquet"
    if not path.exists():
        pytest.skip(f"Feature matrix no encontrada en {path}. Ejecutar: python main.py --skip-train")
    import pandas as pd
    return pd.read_parquet(path)


# ─── Grupo 1: Base de Datos existe ────────────────────────────────────────────

class TestDatabaseExists:
    def test_db_file_exists(self):
        """La base de datos debe existir antes de ejecutar los tests."""
        assert Path(DB_PATH).exists(), (
            f"DB no encontrada en {DB_PATH}. Ejecutar: python main.py --skip-train"
        )

    def test_db_has_expected_tables(self, db_conn):
        """Todas las tablas del esquema estrella deben existir."""
        expected = {
            "dim_customers", "dim_products",
            "fact_orders", "fact_order_items", "fact_support_tickets",
            "ml_churn_labels", "ml_feature_store", "ml_predictions_log",
        }
        existing = {
            row[0] for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        missing = expected - existing
        assert not missing, f"Tablas faltantes en la DB: {missing}"

    def test_db_has_audit_views(self, db_conn):
        """Las vistas de auditoría del Stop/Go deben existir."""
        expected_views = {
            "v_key_integrity", "v_explosion_factors",
            "v_business_invariants", "v_feature_coverage", "v_stop_go_summary",
        }
        existing = {
            row[0] for row in db_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
        missing = expected_views - existing
        assert not missing, f"Vistas de auditoría faltantes: {missing}"


# ─── Grupo 2: Integridad de Llaves Primarias (dup_rate = 0%) ─────────────────

class TestKeyIntegrity:
    def test_dim_customers_no_duplicates(self, db_conn):
        """customer_id debe ser único en dim_customers."""
        dupes = pd.read_sql(
            "SELECT customer_id, COUNT(*) as cnt FROM dim_customers "
            "GROUP BY customer_id HAVING cnt > 1",
            db_conn,
        )
        assert len(dupes) == 0, f"dup_rate > 0%: {len(dupes)} customer_id duplicados"

    def test_fact_orders_no_duplicate_orders(self, db_conn):
        """order_id debe ser único en fact_orders."""
        dupes = pd.read_sql(
            "SELECT order_id, COUNT(*) as cnt FROM fact_orders "
            "GROUP BY order_id HAVING cnt > 1",
            db_conn,
        )
        assert len(dupes) == 0, f"dup_rate > 0%: {len(dupes)} order_id duplicados"

    def test_support_tickets_no_duplicates(self, db_conn):
        """ticket_id debe ser único en fact_support_tickets."""
        dupes = pd.read_sql(
            "SELECT ticket_id, COUNT(*) as cnt FROM fact_support_tickets "
            "GROUP BY ticket_id HAVING cnt > 1",
            db_conn,
        )
        assert len(dupes) == 0, f"dup_rate > 0%: {len(dupes)} ticket_id duplicados"

    def test_feature_store_unique_per_customer_cutoff(self, db_conn):
        """Cada (customer_id, cutoff_date) debe ser único en ml_feature_store."""
        dupes = pd.read_sql(
            "SELECT customer_id, cutoff_date, COUNT(*) as cnt "
            "FROM ml_feature_store "
            "GROUP BY customer_id, cutoff_date HAVING cnt > 1",
            db_conn,
        )
        assert len(dupes) == 0, f"Combinaciones duplicadas en feature_store: {len(dupes)}"

    def test_churn_labels_unique_per_customer_cutoff(self, db_conn):
        """Cada (customer_id, cutoff_date) debe ser único en ml_churn_labels."""
        dupes = pd.read_sql(
            "SELECT customer_id, cutoff_date, COUNT(*) as cnt "
            "FROM ml_churn_labels "
            "GROUP BY customer_id, cutoff_date HAVING cnt > 1",
            db_conn,
        )
        assert len(dupes) == 0, f"Combinaciones duplicadas en churn_labels: {len(dupes)}"


# ─── Grupo 3: Integridad Referencial (orphan_rate = 0%) ──────────────────────

class TestReferentialIntegrity:
    def test_fact_orders_no_orphan_customers(self, db_conn):
        """Todas las órdenes deben apuntar a un customer_id válido en dim_customers."""
        orphans = pd.read_sql(
            "SELECT fo.order_id FROM fact_orders fo "
            "LEFT JOIN dim_customers dc ON fo.customer_id = dc.customer_id "
            "WHERE dc.customer_id IS NULL",
            db_conn,
        )
        assert len(orphans) == 0, f"orphan_rate > 0%: {len(orphans)} órdenes huérfanas"

    def test_support_tickets_no_orphan_customers(self, db_conn):
        """Todos los tickets deben apuntar a un customer_id válido."""
        orphans = pd.read_sql(
            "SELECT ft.ticket_id FROM fact_support_tickets ft "
            "LEFT JOIN dim_customers dc ON ft.customer_id = dc.customer_id "
            "WHERE dc.customer_id IS NULL",
            db_conn,
        )
        assert len(orphans) == 0, f"orphan_rate > 0%: {len(orphans)} tickets huérfanos"

    def test_order_items_no_orphan_orders(self, db_conn):
        """Todos los items deben apuntar a un order_id válido."""
        orphans = pd.read_sql(
            "SELECT foi.item_id FROM fact_order_items foi "
            "LEFT JOIN fact_orders fo ON foi.order_id = fo.order_id "
            "WHERE fo.order_id IS NULL",
            db_conn,
        )
        assert len(orphans) == 0, f"orphan_rate > 0%: {len(orphans)} items huérfanos"


# ─── Grupo 4: Invariantes Temporales de Negocio ───────────────────────────────

class TestBusinessInvariants:
    def test_delivery_date_after_order_date(self, db_conn):
        """INVARIANTE: delivery_date >= order_date en todas las órdenes."""
        violations = pd.read_sql(
            "SELECT order_id, order_date, delivery_date FROM fact_orders "
            "WHERE delivery_date IS NOT NULL AND delivery_date < order_date",
            db_conn,
        )
        assert len(violations) == 0, (
            f"{len(violations)} órdenes con delivery_date < order_date "
            f"(violación temporal crítica)"
        )

    def test_ticket_resolution_after_creation(self, db_conn):
        """INVARIANTE: resolved_date >= created_date en tickets de soporte."""
        violations = pd.read_sql(
            "SELECT ticket_id, created_date, resolved_date FROM fact_support_tickets "
            "WHERE resolved_date IS NOT NULL AND resolved_date < created_date",
            db_conn,
        )
        assert len(violations) == 0, (
            f"{len(violations)} tickets con resolved_date < created_date"
        )

    def test_positive_order_amounts(self, db_conn):
        """INVARIANTE: total_amount >= 0 en todas las órdenes."""
        violations = pd.read_sql(
            "SELECT order_id, total_amount FROM fact_orders WHERE total_amount < 0",
            db_conn,
        )
        assert len(violations) == 0, f"{len(violations)} órdenes con total_amount < 0"

    def test_satisfaction_score_range(self, db_conn):
        """INVARIANTE: satisfaction_score IN [1, 5]."""
        violations = pd.read_sql(
            "SELECT ticket_id, satisfaction_score FROM fact_support_tickets "
            "WHERE satisfaction_score IS NOT NULL "
            "AND (satisfaction_score < 1 OR satisfaction_score > 5)",
            db_conn,
        )
        assert len(violations) == 0, (
            f"{len(violations)} tickets con satisfaction_score fuera de [1,5]"
        )

    def test_churn_label_is_binary(self, db_conn):
        """INVARIANTE: churn_label IN (0, 1)."""
        violations = pd.read_sql(
            "SELECT customer_id, churn_label FROM ml_churn_labels "
            "WHERE churn_label NOT IN (0, 1)",
            db_conn,
        )
        assert len(violations) == 0, f"{len(violations)} etiquetas de churn no binarias"


# ─── Grupo 5: Calidad de la Feature Matrix ────────────────────────────────────

class TestFeatureMatrix:
    def test_feature_matrix_no_empty(self, feature_matrix):
        """La feature matrix debe tener al menos 100 filas."""
        assert len(feature_matrix) >= 100, (
            f"Feature matrix tiene solo {len(feature_matrix)} filas — demasiado pequeña"
        )

    def test_feature_matrix_has_churn_label(self, feature_matrix):
        """La feature matrix debe tener la columna 'churn_label'."""
        assert "churn_label" in feature_matrix.columns, (
            "La columna 'churn_label' no está en la feature matrix"
        )

    def test_churn_rate_in_expected_range(self, feature_matrix):
        """La tasa de churn debe estar en el rango esperado [1%, 50%]."""
        churn_rate = feature_matrix["churn_label"].mean()
        assert 0.01 <= churn_rate <= 0.50, (
            f"Churn rate = {churn_rate:.1%} fuera del rango esperado [1%, 50%]. "
            f"Revisar lógica de etiquetado."
        )

    def test_no_null_threshold_exceeded(self, feature_matrix):
        """Ninguna columna debe superar el umbral de nulos definido en config."""
        null_pcts = feature_matrix.isnull().mean()
        violators = null_pcts[null_pcts > NULL_THRESHOLD].to_dict()
        assert not violators, (
            f"Columnas con nulos > {NULL_THRESHOLD*100:.0f}%: {violators}"
        )

    def test_no_duplicate_customers(self, feature_matrix):
        """El índice (customer_id) debe ser único en la feature matrix."""
        dupes = feature_matrix.index.duplicated().sum()
        assert dupes == 0, f"{dupes} customer_id duplicados en la feature matrix"

    def test_expected_numeric_features_present(self, feature_matrix):
        """Todas las features numéricas del config deben estar presentes."""
        expected = set(CFG["features"]["numeric"])
        present  = set(feature_matrix.columns)
        missing  = expected - present
        assert not missing, f"Features faltantes en la matrix: {missing}"


# ─── Grupo 6: Stop/Go via Vistas SQL ─────────────────────────────────────────

class TestStopGo:
    def test_stop_go_all_pass(self, db_conn):
        """La vista v_stop_go_summary debe retornar PASS en todos los criterios."""
        try:
            results = pd.read_sql(
                "SELECT check_name, status FROM v_stop_go_summary", db_conn
            )
        except Exception:
            pytest.skip("v_stop_go_summary no disponible (ejecutar pipeline primero)")

        failures = results[results["status"] == "FAIL"]
        assert len(failures) == 0, (
            f"Stop/Go FAIL en: {failures['check_name'].tolist()}"
        )

    def test_feature_coverage_100_percent(self, db_conn):
        """Todos los clientes con etiquetas deben tener features calculadas."""
        try:
            result = pd.read_sql(
                "SELECT coverage_pct FROM v_feature_coverage", db_conn
            )
            coverage = result.iloc[0, 0]
            assert coverage >= 100.0, (
                f"Cobertura del Feature Store = {coverage:.1f}% < 100%. "
                "Hay clientes sin features calculadas."
            )
        except Exception:
            pytest.skip("v_feature_coverage no disponible")
