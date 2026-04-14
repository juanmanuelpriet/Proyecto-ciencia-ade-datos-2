"""
sql/02_seed_data.py — Generador de Datos Sintéticos Realistas
=============================================================
Genera ~5,000 clientes y ~120,000 transacciones con ruido controlado,
valores nulos, y desbalanceo de clases natural (~15% churn).

DECISIÓN: Los datos sintéticos siguen distribuciones de negocio reales:
- Distribución de Pareto para gasto (80% ingresos = 20% clientes)
- Distribución de Poisson para frecuencia de compras
- Tasa de churn dependiente de recency (correlación realista)
"""

import sqlite3
import uuid
import random
import json
import hashlib
import os
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import yaml
from loguru import logger


# ─── Carga de Configuración ───────────────────────────────────────────────────
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DB_PATH      = CFG["paths"]["db"]
RANDOM_SEED  = CFG["project"]["random_seed"]
CUTOFF_DATE  = date.fromisoformat(CFG["temporal"]["cutoff_date"])
OBS_WINDOW   = CFG["temporal"]["observation_window_days"]
PRED_WINDOW  = CFG["temporal"]["prediction_window_days"]

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── Constantes del dominio ─────────────────────────────────────────────────────
N_CUSTOMERS  = 5_000
CATEGORIES   = ["Electronics", "Clothing", "Home", "Sports", "Beauty", "Books", "Food"]
SEGMENTS     = ["VIP", "STANDARD", "NEW", "AT_RISK"]
SEG_WEIGHTS  = [0.10, 0.60, 0.20, 0.10]
COUNTRIES    = ["CO", "MX", "PE", "AR", "CL"]
STATUSES     = ["COMPLETED", "CANCELLED", "RETURNED", "FAILED"]
STATUS_PROBS = [0.80, 0.08, 0.08, 0.04]
CHANNELS     = ["WEB", "APP", "PHONE"]
TICKET_TYPES = ["DELIVERY_ISSUE", "PRODUCT_DEFECT", "REFUND", "BILLING", "OTHER"]
PAYMENT_MET  = ["CREDIT_CARD", "PSE", "COD", "DIGITAL_WALLET"]


def _generate_customers() -> list[dict]:
    """Genera el maestro de clientes con distribución realista de segmentos."""
    customers = []
    segments = rng.choice(SEGMENTS, size=N_CUSTOMERS, p=SEG_WEIGHTS)

    for i in range(N_CUSTOMERS):
        seg = segments[i]
        # Fecha de registro: entre 2019 y 2023 (clientes maduros)
        days_ago = int(rng.integers(30, 1500))
        reg_date = CUTOFF_DATE - timedelta(days=days_ago)

        customers.append({
            "customer_id":       f"C{str(i+1).zfill(5)}",
            "registration_date": reg_date.isoformat(),
            "country":           random.choice(COUNTRIES),
            "customer_segment":  seg,
            "preferred_category": random.choice(CATEGORIES),
            # ~10% han hecho opt-out del email → señal negativa
            "email_opt_in":      int(rng.random() > 0.10),
        })
    return customers


def _generate_products(n: int = 500) -> list[dict]:
    """Genera catálogo de productos con distribución de precios por categoría."""
    products = []
    price_ranges = {
        "Electronics": (50, 1500),
        "Clothing":     (15, 200),
        "Home":         (20, 500),
        "Sports":       (10, 300),
        "Beauty":       (5, 120),
        "Books":        (8, 60),
        "Food":         (3, 80),
    }
    for i in range(n):
        cat      = random.choice(CATEGORIES)
        lo, hi   = price_ranges[cat]
        price    = round(float(rng.uniform(lo, hi)), 2)
        products.append({
            "product_id":   f"P{str(i+1).zfill(4)}",
            "product_name": f"{cat}_Product_{i+1}",
            "category":     cat,
            "subcategory":  f"{cat}_Sub_{(i % 5) + 1}",
            "unit_price":   price,
            "unit_cost":    round(price * float(rng.uniform(0.4, 0.7)), 2),
            "is_active":    int(rng.random() > 0.03),  # 3% de productos inactivos
        })
    return products


def _generate_orders_and_labels(
    customers: list[dict],
    products: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Genera órdenes, line items, tickets y labels de churn.

    LÓGICA DE CHURN (realista):
    - Clientes VIP tienen < 5% de probabilidad de churn
    - Segmento AT_RISK tiene ~60% de probabilidad de churn
    - Clientes inactivos > 60 días tienen probabilidad de churn incrementada
    - Clientes con más de 2 tickets de soporte → más propensos al churn
    """
    churn_base = {"VIP": 0.05, "STANDARD": 0.15, "NEW": 0.25, "AT_RISK": 0.60}

    orders          = []
    order_items     = []
    support_tickets = []
    churn_labels    = []

    product_df = pd.DataFrame(products)
    item_counter = 1

    obs_start_global = CUTOFF_DATE - timedelta(days=OBS_WINDOW)
    pred_end_global  = CUTOFF_DATE + timedelta(days=PRED_WINDOW)

    for cust in customers:
        cid  = cust["customer_id"]
        seg  = cust["customer_segment"]
        reg  = date.fromisoformat(cust["registration_date"])

        # Cuántas órdenes en el periodo total (registro hasta pred_end)
        total_days = (pred_end_global - reg).days
        if total_days <= 0:
            continue

        # Frecuencia de compra mensual según segmento (Poisson)
        lam_map = {"VIP": 4.0, "STANDARD": 1.5, "NEW": 0.8, "AT_RISK": 0.5}
        lam_monthly = lam_map[seg]
        n_orders = max(0, int(rng.poisson(lam_monthly * (total_days / 30))))

        # Genera fechas de órdenes aleatorias en el rango válido
        order_dates_offsets = sorted(rng.integers(0, total_days, size=n_orders))
        order_dates = [reg + timedelta(days=int(d)) for d in order_dates_offsets]

        # Genera tickets de soporte (1-2 por cada 10 órdenes en promedio)
        n_tickets = int(rng.poisson(max(0.1, n_orders * 0.12)))
        for t in range(n_tickets):
            t_date = reg + timedelta(days=int(rng.integers(0, max(1, total_days))))
            t_sat = None if rng.random() < 0.30 else int(rng.integers(1, 6))  # 30% no responden
            r_date = (t_date + timedelta(days=int(rng.integers(1, 10))))
            r_date = r_date if r_date <= pred_end_global else None
            support_tickets.append({
                "ticket_id":         f"T{cid}_{t}",
                "customer_id":       cid,
                "created_date":      t_date.isoformat(),
                "resolved_date":     r_date.isoformat() if r_date else None,
                "ticket_type":       random.choice(TICKET_TYPES),
                "priority":          random.choice(["LOW", "MEDIUM", "HIGH"]),
                "resolution_status": random.choice(["RESOLVED", "OPEN", "ESCALATED"]),
                "satisfaction_score": t_sat,
            })

        # Alta de tickets incrementa la probabilidad de churn
        base_churn_p = churn_base[seg]
        if n_tickets > 3:
            base_churn_p = min(0.95, base_churn_p + 0.15)

        # Registra órdenes en base de datos
        obs_orders = [d for d in order_dates if obs_start_global <= d <= CUTOFF_DATE]
        pred_orders = [d for d in order_dates if CUTOFF_DATE < d <= pred_end_global]

        for od in order_dates:
            status = rng.choice(STATUSES, p=STATUS_PROBS)
            # Gasto según segmento (Pareto: VIPs gastan mucho más)
            spend_map = {"VIP": (200, 1500), "STANDARD": (30, 300), "NEW": (20, 150), "AT_RISK": (10, 100)}
            lo_s, hi_s = spend_map[seg]
            total_amt = round(float(rng.uniform(lo_s, hi_s)), 2)
            discount  = round(total_amt * float(rng.uniform(0, 0.20)), 2)
            del_days  = int(rng.integers(1, 15)) if status == "COMPLETED" else None
            del_date  = (od + timedelta(days=del_days)).isoformat() if del_days else None

            oid = f"ORD_{cid}_{od.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
            orders.append({
                "order_id":         oid,
                "customer_id":      cid,
                "order_date":       od.isoformat(),
                "delivery_date":    del_date,
                "order_status":     status,
                "total_amount":     total_amt,
                "discount_applied": discount,
                "payment_method":   random.choice(PAYMENT_MET),
                "channel":          random.choice(CHANNELS),
            })

            # Line items (1 a 4 productos por orden)
            n_items = int(rng.integers(1, 5))
            sampled = product_df.sample(n=min(n_items, len(product_df)), random_state=None)
            for _, prod_row in sampled.iterrows():
                qty = int(rng.integers(1, 4))
                # Precio puede tener hasta 5% de variación respecto al catálogo (ruido real)
                cat_price = prod_row["unit_price"] * float(rng.uniform(0.95, 1.05))
                order_items.append({
                    "item_id":    item_counter,
                    "order_id":   oid,
                    "product_id": prod_row["product_id"],
                    "quantity":   qty,
                    "unit_price": round(cat_price, 2),
                })
                item_counter += 1

        # ── Calcular Label de Churn ────────────────────────────────────────
        # Solo etiquetamos clientes con al menos 1 orden en ventana observación
        if len(obs_orders) == 0:
            continue

        # La probabilidad de churn se hace más alta si el cliente no compró en últimos 60d
        last_buy = max(obs_orders) if obs_orders else None
        if last_buy and (CUTOFF_DATE - last_buy).days > 60:
            base_churn_p = min(0.95, base_churn_p + 0.25)

        did_churn = int(rng.random() < base_churn_p) if len(pred_orders) == 0 else 0
        if len(pred_orders) == 0:
            did_churn = int(rng.random() < base_churn_p)
        else:
            # Si hay órdenes en pred window, el cliente NO hizo churn (definición)
            did_churn = 0

        churn_labels.append({
            "customer_id":         cid,
            "cutoff_date":         CUTOFF_DATE.isoformat(),
            "observation_start":   obs_start_global.isoformat(),
            "prediction_end":      pred_end_global.isoformat(),
            "churn_label":         did_churn,
            "orders_in_obs_window":  len(obs_orders),
            "orders_in_pred_window": len(pred_orders),
            "last_order_date":     max(obs_orders).isoformat() if obs_orders else None,
        })

    return orders, order_items, support_tickets, churn_labels


def _bulk_insert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    """Inserta filas en bulk usando executemany (mucho más rápido que row-by-row)."""
    if not rows:
        return
    cols        = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(cols))
    sql         = f"INSERT OR IGNORE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    data        = [tuple(row[c] for c in cols) for row in rows]
    conn.executemany(sql, data)


def seed(force: bool = False) -> None:
    """
    Punto de entrada principal. Genera y persiste los datos en SQLite.
    Si la BD ya existe y force=False, no regenera (idempotente).
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    if os.path.exists(DB_PATH) and not force:
        logger.info(f"Base de datos ya existe en {DB_PATH}. Skipping seed. (usa force=True para regenerar)")
        return

    logger.info("🌱 Iniciando generación de datos sintéticos...")

    # ── Crear esquema ──────────────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    with open("sql/01_ddl_schema.sql") as f:
        conn.executescript(f.read())
    conn.commit()
    logger.info("✅ Esquema DDL creado.")

    # ── Generar y cargar datos ─────────────────────────────────────────────────
    customers = _generate_customers()
    _bulk_insert(conn, "dim_customers", customers)
    conn.commit()
    logger.info(f"✅ {len(customers)} clientes insertados.")

    products = _generate_products()
    _bulk_insert(conn, "dim_products", products)
    conn.commit()
    logger.info(f"✅ {len(products)} productos insertados.")

    orders, items, tickets, labels = _generate_orders_and_labels(customers, products)
    _bulk_insert(conn, "fact_orders", orders)
    conn.commit()
    logger.info(f"✅ {len(orders)} órdenes insertadas.")

    _bulk_insert(conn, "fact_order_items", items)
    conn.commit()
    logger.info(f"✅ {len(items)} line items insertados.")

    _bulk_insert(conn, "fact_support_tickets", tickets)
    conn.commit()
    logger.info(f"✅ {len(tickets)} tickets de soporte insertados.")

    _bulk_insert(conn, "ml_churn_labels", labels)
    conn.commit()

    # ── Estadísticas de etiquetas ──────────────────────────────────────────────
    n_churn    = sum(r["churn_label"] for r in labels)
    n_no_churn = len(labels) - n_churn
    churn_rate = n_churn / len(labels) if labels else 0
    logger.info(
        f"✅ {len(labels)} labels calculados | "
        f"Churn: {n_churn} ({churn_rate:.1%}) | No-Churn: {n_no_churn}"
    )

    # ── Guardar hash de los datos para reproducibilidad (manifest) ─────────────
    data_hash = hashlib.md5(
        json.dumps({"n_customers": len(customers), "n_orders": len(orders),
                    "n_labels": len(labels), "seed": RANDOM_SEED}).encode()
    ).hexdigest()
    logger.info(f"📋 Data hash: {data_hash}")

    conn.close()
    logger.info(f"🚀 Seed completado. DB guardada en: {DB_PATH}")


if __name__ == "__main__":
    seed(force=False)
