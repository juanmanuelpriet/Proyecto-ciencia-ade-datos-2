-- =============================================================================
-- MÓDULO 1: DATA FOUNDATION — DDL del Sistema de Churn
-- Base de datos: SQLite (desarrollo) / PostgreSQL (producción)
-- Autor: Sistema E2E Churn Predictor
-- =============================================================================
-- DECISIÓN DE DISEÑO: Se separa en dimensiones (dim_*), hechos (fact_*) y
-- tablas de ML (ml_*). Esto permite que el modelo vea solo features
-- calculadas al momento de la predicción (point-in-time correctness).
-- =============================================================================

PRAGMA journal_mode = WAL;      -- Write-Ahead Logging: mejor concurrencia lectura/escritura
PRAGMA foreign_keys = ON;       -- Integridad referencial siempre activa

-- ---------------------------------------------------------------------------
-- DIMENSIÓN: Clientes
-- Maestro de clientes. Datos que no cambian o cambian muy poco.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_customers (
    customer_id         TEXT PRIMARY KEY,       -- UUID o código de negocio
    registration_date   DATE NOT NULL,
    country             TEXT NOT NULL DEFAULT 'CO',
    customer_segment    TEXT DEFAULT 'STANDARD',    -- e.g., VIP, STANDARD, NEW
    preferred_category  TEXT,                       -- Categoría más comprada (pre-calculada)
    email_opt_in        INTEGER DEFAULT 1,           -- 1 = activo, 0 = desuscrito (señal de churn)
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Validación de integridad en inserción
    CONSTRAINT chk_segment CHECK (customer_segment IN ('VIP', 'STANDARD', 'NEW', 'AT_RISK')),
    CONSTRAINT chk_email_opt CHECK (email_opt_in IN (0, 1))
);

-- Índice en fecha de registro para filtros de cohorte
CREATE INDEX IF NOT EXISTS idx_dim_customers_reg_date ON dim_customers(registration_date);

-- ---------------------------------------------------------------------------
-- DIMENSIÓN: Productos / Catálogo
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_products (
    product_id      TEXT PRIMARY KEY,
    product_name    TEXT NOT NULL,
    category        TEXT NOT NULL,          -- Categoría principal (Electronics, Clothing, etc.)
    subcategory     TEXT,
    unit_price      REAL NOT NULL,
    unit_cost       REAL,                   -- Para calcular margen (opcional)
    is_active       INTEGER DEFAULT 1,
    CONSTRAINT chk_price CHECK (unit_price > 0)
);

CREATE INDEX IF NOT EXISTS idx_dim_products_category ON dim_products(category);

-- ---------------------------------------------------------------------------
-- HECHO: Órdenes (grain: una fila = un pedido completo)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_orders (
    order_id            TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES dim_customers(customer_id),
    order_date          DATE NOT NULL,
    delivery_date       DATE,               -- NULL si aún no entregado
    order_status        TEXT NOT NULL DEFAULT 'COMPLETED',
    total_amount        REAL NOT NULL,
    discount_applied    REAL DEFAULT 0.0,   -- Monto de descuento en $
    payment_method      TEXT,               -- CREDIT_CARD, PSE, COD, etc.
    channel             TEXT DEFAULT 'WEB', -- WEB, APP, PHONE
    -- Invariante de negocio: delivery_date debe ser >= order_date
    CONSTRAINT chk_delivery_after_order
        CHECK (delivery_date IS NULL OR delivery_date >= order_date),
    CONSTRAINT chk_total_positive CHECK (total_amount >= 0),
    CONSTRAINT chk_discount_non_negative CHECK (discount_applied >= 0),
    CONSTRAINT chk_status CHECK (order_status IN
        ('COMPLETED', 'CANCELLED', 'RETURNED', 'PENDING', 'FAILED'))
);

-- CRÍTICO: Índice compuesto (customer_id, order_date) para ventanas temporales
-- Esta es la consulta más frecuente del feature engineering
CREATE INDEX IF NOT EXISTS idx_fact_orders_cust_date
    ON fact_orders(customer_id, order_date);
CREATE INDEX IF NOT EXISTS idx_fact_orders_date ON fact_orders(order_date);
CREATE INDEX IF NOT EXISTS idx_fact_orders_status ON fact_orders(order_status);

-- ---------------------------------------------------------------------------
-- HECHO: Líneas de Orden (grain: producto dentro de un pedido)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_order_items (
    item_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT NOT NULL REFERENCES fact_orders(order_id),
    product_id      TEXT NOT NULL REFERENCES dim_products(product_id),
    quantity        INTEGER NOT NULL DEFAULT 1,
    unit_price      REAL NOT NULL,          -- Precio al momento de la compra (puede diferir del catálogo)
    CONSTRAINT chk_qty_positive CHECK (quantity > 0),
    CONSTRAINT chk_item_price CHECK (unit_price >= 0)
);

CREATE INDEX IF NOT EXISTS idx_fact_items_order ON fact_order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_fact_items_product ON fact_order_items(product_id);

-- ---------------------------------------------------------------------------
-- HECHO: Tickets de Soporte (señal clave de insatisfacción → churn)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_support_tickets (
    ticket_id           TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES dim_customers(customer_id),
    created_date        DATE NOT NULL,
    resolved_date       DATE,
    ticket_type         TEXT NOT NULL,      -- DELIVERY_ISSUE, PRODUCT_DEFECT, REFUND, BILLING, OTHER
    priority            TEXT DEFAULT 'MEDIUM',
    resolution_status   TEXT DEFAULT 'OPEN', -- OPEN, RESOLVED, ESCALATED, ABANDONED
    satisfaction_score  INTEGER,            -- NPS/CSAT 1-5 (puede ser NULL si no respondió)
    CONSTRAINT chk_ticket_type CHECK (ticket_type IN
        ('DELIVERY_ISSUE', 'PRODUCT_DEFECT', 'REFUND', 'BILLING', 'OTHER')),
    CONSTRAINT chk_satisfaction CHECK (
        satisfaction_score IS NULL OR
        satisfaction_score BETWEEN 1 AND 5),
    CONSTRAINT chk_resolved_after_created CHECK (
        resolved_date IS NULL OR resolved_date >= created_date)
);

CREATE INDEX IF NOT EXISTS idx_support_cust_date
    ON fact_support_tickets(customer_id, created_date);

-- ---------------------------------------------------------------------------
-- ML: Etiquetas de Churn (TARGET — calculado con lógica temporal rigurosa)
--
-- DEFINICIÓN de Churn: Un cliente hace CHURN en el periodo T si en los
-- próximos 30 días después de la fecha de corte NO realiza ningún pedido,
-- Y tuvo al menos una compra en los 90 días previos a la fecha de corte.
-- Esto evita etiquetar como "churn" a clientes que nunca fueron activos.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_churn_labels (
    label_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         TEXT NOT NULL REFERENCES dim_customers(customer_id),
    cutoff_date         DATE NOT NULL,          -- Fecha de corte (no hay info más allá de aquí)
    observation_start   DATE NOT NULL,          -- cutoff_date - 90 días
    prediction_end      DATE NOT NULL,          -- cutoff_date + 30 días
    churn_label         INTEGER NOT NULL,       -- 0 = retuvo, 1 = churned
    -- Evidencia para auditar la etiqueta
    orders_in_obs_window    INTEGER,            -- Órdenes en los 90 días de observación
    orders_in_pred_window   INTEGER,            -- Órdenes en los 30 días de predicción
    last_order_date         DATE,               -- Última orden antes del cutoff
    label_created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_churn_binary CHECK (churn_label IN (0, 1)),
    CONSTRAINT uq_customer_cutoff UNIQUE (customer_id, cutoff_date)
);

CREATE INDEX IF NOT EXISTS idx_churn_labels_customer ON ml_churn_labels(customer_id);
CREATE INDEX IF NOT EXISTS idx_churn_labels_label ON ml_churn_labels(churn_label);

-- ---------------------------------------------------------------------------
-- ML: Feature Store (features calculadas point-in-time, listas para el modelo)
-- Esta tabla separa el feature engineering del entrenamiento,
-- permitiendo reusar features para predicciones en batch (producción).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_feature_store (
    feature_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id             TEXT NOT NULL REFERENCES dim_customers(customer_id),
    cutoff_date             DATE NOT NULL,

    -- === RFM CLÁSICO ===
    recency_days            REAL,       -- Días desde última compra hasta cutoff
    frequency_orders        INTEGER,    -- Número de órdenes en ventana 90d
    monetary_total          REAL,       -- Gasto total en ventana 90d
    monetary_avg_order      REAL,       -- Gasto promedio por orden

    -- === SEÑALES DE TENDENCIA ===
    -- Pendiente de regresión lineal del gasto en últimos 30d vs previos 60d
    -- Negativo = gasto cayendo → señal de churn
    trend_slope_30d         REAL,

    -- === DIVERSIDAD DE COMPRA ===
    category_diversity      INTEGER,    -- Nro. de categorías distintas compradas

    -- === SEÑALES DE FRICCIÓN (soporte) ===
    support_tickets_90d     INTEGER,    -- Tickets abiertos en la ventana
    avg_satisfaction_score  REAL,       -- Promedio de satisfaction_score (puede ser NULL)

    -- === LOGÍSTICA ===
    avg_delivery_days       REAL,       -- Días promedio de entrega (fricción operacional)
    return_rate             REAL,       -- Tasa de devoluciones (órdenes_devueltas / total_órdenes)

    -- === TENURE ===
    days_since_registration REAL,       -- Edad del cliente en días

    -- === FEATURES CATEGÓRICAS (se one-hot encodean en el pipeline) ===
    preferred_category      TEXT,
    customer_segment        TEXT,

    -- Metadatos de linaje
    feature_created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pipeline_version        TEXT,       -- Versión del pipeline que generó las features

    CONSTRAINT uq_feature_customer_cutoff UNIQUE (customer_id, cutoff_date)
);

CREATE INDEX IF NOT EXISTS idx_feature_store_customer ON ml_feature_store(customer_id);
CREATE INDEX IF NOT EXISTS idx_feature_store_cutoff ON ml_feature_store(cutoff_date);

-- ---------------------------------------------------------------------------
-- AUDIT: Registro de predicciones del modelo en producción
-- Permite monitoreo de drift y recalibración periódica
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ml_predictions_log (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id         TEXT NOT NULL,
    prediction_date     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version       TEXT NOT NULL,
    churn_probability   REAL NOT NULL,
    churn_flag          INTEGER NOT NULL,    -- 0 o 1, según umbral optimizado
    decision_threshold  REAL NOT NULL,
    action_triggered    TEXT,               -- 'COUPON_SENT', 'CALL_SCHEDULED', 'NO_ACTION'
    CONSTRAINT chk_prob_range CHECK (churn_probability BETWEEN 0.0 AND 1.0),
    CONSTRAINT chk_flag CHECK (churn_flag IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_pred_log_customer ON ml_predictions_log(customer_id);
CREATE INDEX IF NOT EXISTS idx_pred_log_date ON ml_predictions_log(prediction_date);

-- =============================================================================
-- VISTAS DE AUDITORÍA Y CALIDAD DE DATOS (Fase III)
-- Estas vistas implementan los chequeos de la Regla Stop/Go directamente
-- en la base de datos, permitiendo auditoría sin ejecutar el pipeline Python.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- VISTA: Resumen de integridad de llaves por tabla
-- Reporta: total, duplicados, orphan rate
-- Satisface: Fase III — Integridad de Llaves (dup_rate, orphan_rate)
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_key_integrity AS
SELECT
    'dim_customers'      AS table_name,
    COUNT(*)             AS total_rows,
    COUNT(*) - COUNT(DISTINCT customer_id) AS duplicate_keys,
    ROUND((COUNT(*) - COUNT(DISTINCT customer_id)) * 100.0 / MAX(COUNT(*), 1), 4) AS dup_rate_pct,
    NULL                 AS orphan_rows,
    NULL                 AS orphan_rate_pct
FROM dim_customers
UNION ALL
SELECT
    'fact_orders',
    COUNT(*),
    COUNT(*) - COUNT(DISTINCT order_id),
    ROUND((COUNT(*) - COUNT(DISTINCT order_id)) * 100.0 / MAX(COUNT(*), 1), 4),
    SUM(CASE WHEN dc.customer_id IS NULL THEN 1 ELSE 0 END),
    ROUND(SUM(CASE WHEN dc.customer_id IS NULL THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1), 4)
FROM fact_orders fo
LEFT JOIN dim_customers dc ON fo.customer_id = dc.customer_id
UNION ALL
SELECT
    'fact_support_tickets',
    COUNT(*),
    COUNT(*) - COUNT(DISTINCT ticket_id),
    ROUND((COUNT(*) - COUNT(DISTINCT ticket_id)) * 100.0 / MAX(COUNT(*), 1), 4),
    SUM(CASE WHEN dc.customer_id IS NULL THEN 1 ELSE 0 END),
    ROUND(SUM(CASE WHEN dc.customer_id IS NULL THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1), 4)
FROM fact_support_tickets ft
LEFT JOIN dim_customers dc ON ft.customer_id = dc.customer_id
UNION ALL
SELECT
    'fact_order_items',
    COUNT(*),
    COUNT(*) - COUNT(DISTINCT item_id),
    ROUND((COUNT(*) - COUNT(DISTINCT item_id)) * 100.0 / MAX(COUNT(*), 1), 4),
    SUM(CASE WHEN fo.order_id IS NULL THEN 1 ELSE 0 END),
    ROUND(SUM(CASE WHEN fo.order_id IS NULL THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1), 4)
FROM fact_order_items foi
LEFT JOIN fact_orders fo ON foi.order_id = fo.order_id;

-- ---------------------------------------------------------------------------
-- VISTA: Factor de explosión de joins (Fase II — Riesgo de Join)
-- explosion(A, B) = |A ⋈ B| / |A|
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_explosion_factors AS
SELECT
    'dim_customers ⋈ fact_orders'           AS join_name,
    (SELECT COUNT(*) FROM dim_customers)     AS n_left,
    (SELECT COUNT(*) FROM fact_orders)       AS n_right,
    ROUND(
        CAST((SELECT COUNT(*) FROM fact_orders) AS REAL) /
        MAX((SELECT COUNT(*) FROM dim_customers), 1),
        3
    ) AS explosion_factor
UNION ALL
SELECT
    'fact_orders ⋈ fact_order_items',
    (SELECT COUNT(*) FROM fact_orders),
    (SELECT COUNT(*) FROM fact_order_items),
    ROUND(
        CAST((SELECT COUNT(*) FROM fact_order_items) AS REAL) /
        MAX((SELECT COUNT(*) FROM fact_orders), 1),
        3
    )
UNION ALL
SELECT
    'dim_customers ⋈ fact_support_tickets',
    (SELECT COUNT(*) FROM dim_customers),
    (SELECT COUNT(*) FROM fact_support_tickets),
    ROUND(
        CAST((SELECT COUNT(*) FROM fact_support_tickets) AS REAL) /
        MAX((SELECT COUNT(*) FROM dim_customers), 1),
        3
    );

-- ---------------------------------------------------------------------------
-- VISTA: Validación de invariantes de negocio (Fase II — Protocolo de Acceso)
-- Reporta violaciones de las reglas de negocio implementadas como CHECK
-- constraints en el DDL. Una vista separada permite auditarlas fácilmente.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_business_invariants AS
SELECT
    'delivery_date >= order_date'           AS invariant_name,
    COUNT(*)                                AS total_checked,
    SUM(CASE WHEN delivery_date < order_date THEN 1 ELSE 0 END) AS violations,
    ROUND(
        SUM(CASE WHEN delivery_date < order_date THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1),
        4
    ) AS violation_rate_pct
FROM fact_orders
WHERE delivery_date IS NOT NULL
UNION ALL
SELECT
    'resolved_date >= created_date (support)',
    COUNT(*),
    SUM(CASE WHEN resolved_date < created_date THEN 1 ELSE 0 END),
    ROUND(
        SUM(CASE WHEN resolved_date < created_date THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1),
        4
    )
FROM fact_support_tickets
WHERE resolved_date IS NOT NULL
UNION ALL
SELECT
    'total_amount >= 0 (orders)',
    COUNT(*),
    SUM(CASE WHEN total_amount < 0 THEN 1 ELSE 0 END),
    ROUND(
        SUM(CASE WHEN total_amount < 0 THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1),
        4
    )
FROM fact_orders;

-- ---------------------------------------------------------------------------
-- VISTA: Cobertura del Feature Store (Regla Stop/Go — Cobertura de Joins)
-- Verifica que todos los clientes en ml_churn_labels tengan features.
-- Si coverage_pct < 100%, el snapshot debe ser rechazado (STOP).
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_feature_coverage AS
SELECT
    COUNT(DISTINCT cl.customer_id)                            AS customers_with_labels,
    COUNT(DISTINCT fs.customer_id)                            AS customers_with_features,
    COUNT(DISTINCT cl.customer_id) - COUNT(DISTINCT fs.customer_id) AS customers_without_features,
    ROUND(
        COUNT(DISTINCT fs.customer_id) * 100.0 /
        MAX(COUNT(DISTINCT cl.customer_id), 1),
        2
    ) AS coverage_pct,
    CASE
        WHEN COUNT(DISTINCT fs.customer_id) = COUNT(DISTINCT cl.customer_id)
        THEN 'GO — Cobertura 100%'
        ELSE 'STOP — Clientes sin features'
    END AS stop_go_status
FROM ml_churn_labels cl
LEFT JOIN ml_feature_store fs
    ON cl.customer_id = fs.customer_id
    AND cl.cutoff_date = fs.cutoff_date;

-- ---------------------------------------------------------------------------
-- VISTA: Resumen del Stop/Go completo (Fase III — Regla Stop/Go)
-- Un JOIN de todas las vistas de calidad en una sola consulta.
-- Resultado: 'GO' solo si TODAS las condiciones pasan.
-- ---------------------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_stop_go_summary AS
SELECT
    'Key Integrity'                AS check_name,
    CASE
        WHEN SUM(dup_rate_pct) = 0 AND (SUM(orphan_rate_pct) IS NULL OR SUM(orphan_rate_pct) = 0)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS status,
    'dup_rate=0% en todas las tablas, orphan_rate=0%' AS description
FROM v_key_integrity
UNION ALL
SELECT
    'Business Invariants',
    CASE WHEN SUM(violations) = 0 THEN 'PASS' ELSE 'FAIL' END,
    'Sin violaciones temporales ni de integridad de valores'
FROM v_business_invariants
UNION ALL
SELECT
    'Feature Coverage',
    CASE WHEN coverage_pct = 100 THEN 'PASS' ELSE 'FAIL' END,
    'Todos los clientes con etiquetas tienen features calculadas'
FROM v_feature_coverage;

