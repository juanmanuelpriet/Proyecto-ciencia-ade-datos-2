"""
tests/test_temporal_integrity.py — Verificación de No-Leakage Temporal
=======================================================================
Garantiza que el feature engineering respete estrictamente la fecha de corte t0.
Un fallo aquí invalidaría todo el sistema de predicción.
"""

import pytest
import pandas as pd
from datetime import date, timedelta
from etl.feature_engineer import build_feature_matrix
from etl.extractor import RawDataExtractor

@pytest.fixture
def mock_data():
    """Crea un dataset mínimo para testear leakage."""
    cutoff = date(2023, 1, 1)
    
    customers = pd.DataFrame([{
        "customer_id": "C1", 
        "registration_date": pd.Timestamp("2022-01-01"),
        "customer_segment": "STANDARD",
        "preferred_category": "Home"
    }])
    
    # Órdenes en el pasado y en el FUTURO (leakage potential)
    orders = pd.DataFrame([
        {"order_id": "O1", "customer_id": "C1", "order_date": pd.Timestamp("2022-12-30"), "delivery_date": pd.Timestamp("2023-01-01"), "order_status": "COMPLETED", "total_amount": 100, "discount_applied": 0},
        {"order_id": "O2", "customer_id": "C1", "order_date": pd.Timestamp("2023-01-02"), "delivery_date": pd.Timestamp("2023-01-05"), "order_status": "COMPLETED", "total_amount": 500, "discount_applied": 0}, # FUTURO
    ])
    
    order_items = pd.DataFrame([
        {"order_id": "O1", "product_id": "P1", "category": "Home"},
        {"order_id": "O2", "product_id": "P1", "category": "Home"},
    ])
    
    tickets = pd.DataFrame([
        {"ticket_id": "T1", "customer_id": "C1", "created_date": pd.Timestamp("2022-12-31"), "satisfaction_score": 5},
        {"ticket_id": "T2", "customer_id": "C1", "created_date": pd.Timestamp("2023-01-05"), "satisfaction_score": 1}, # FUTURO
    ])
    
    labels = pd.DataFrame([{
        "customer_id": "C1", "churn_label": 0
    }])
    
    return customers, orders, order_items, tickets, labels, cutoff

def test_feature_engineering_prevents_leakage(mock_data):
    """
    TEST CRÍTICO: Las features calculadas al 2023-01-01 NO deben ver 
    la orden O2 ni el ticket T2.
    """
    customers, orders, order_items, tickets, labels, cutoff = mock_data
    
    # El extractor real filtraría las órdenes, pero build_feature_matrix 
    # es nuestra 'last line of defense' en el cálculo de features.
    # Nota: build_feature_matrix internamente llama a compute_* functions
    
    feat_matrix = build_feature_matrix(
        customers=customers,
        orders=orders,
        order_items=order_items,
        tickets=tickets,
        labels=labels,
        cutoff=cutoff,
        window_days=90
    )
    
    # 1. Verificar Frequency (solo debe ser 1, no 2)
    assert feat_matrix.loc["C1", "frequency_orders"] == 1
    
    # 2. Verificar Monetary Total (solo debe ser 100, no 600)
    assert feat_matrix.loc["C1", "monetary_total"] == 100
    
    # 3. Verificar Tickets (solo debe ser 1, no 2)
    assert feat_matrix.loc["C1", "support_tickets_90d"] == 1
    
    # 4. Verificar Recency (basado en O1 al 30 de dic)
    # 2023-01-01 menos 2022-12-30 = 2 días
    assert feat_matrix.loc["C1", "recency_days"] == 2
