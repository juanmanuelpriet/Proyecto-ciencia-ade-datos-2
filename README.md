# 🛒 Global-E-Shop — Sistema de Predicción de Churn

> **Proyecto de Ciencia de Datos II · Universidad de La Sabana · 2026.1**  
> Docente: Sergio Amortegui

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?logo=xgboost)](https://xgboost.readthedocs.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?logo=sqlite)](https://sqlite.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📌 Descripción

Sistema end-to-end de **predicción de abandono de clientes (Churn)** para la plataforma de e-commerce **Global-E-Shop**. El proyecto transforma datos transaccionales en decisiones de negocio accionables, siguiendo los principios de ciencia de datos **auditable, reproducible y orientada a decisiones**.

**Problema de Negocio:** Identificar clientes con alto riesgo de no volver a comprar en los próximos 30 días, para activar campañas de retención personalizadas con ROI positivo.

**Resultado:** Recall = **0.82** [IC 95%: 0.80, 0.84] · Lift@20% = **2.4x** sobre baseline RFM.

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PIPELINE DAG                                    │
│                                                                       │
│  SQL Seed  ──►  Extract  ──►  Validate  ──►  Clean  ──►  Feature    │
│                                                              │        │
│                                                              ▼        │
│                              FastAPI  ◄──  Model  ◄──  Evaluate     │
└─────────────────────────────────────────────────────────────────────┘
```

### Modelo Temporal (Zero Data Leakage)

```
2023-07-02          2023-09-30 (t₀)       2023-10-30
     │                    │                    │
     ◄────────────────────►────────────────────►
     │  Ventana Observ.   │  Ventana Predic.  │
     │     (90 días)      │    (30 días)      │
     │    [FEATURES]      │    [ETIQUETAS]    │
```

---

## 📁 Estructura del Proyecto

```
proyecto 1/
├── 📄 README.md                          # Este archivo
├── 📄 DESIGN_DOC.md                      # Decisiones arquitectónicas
├── ⚙️  config.yaml                        # Fuente única de verdad (parámetros θ)
├── 🐍 main.py                             # Orquestador principal del pipeline
├── 📋 requirements.txt                    # Dependencias fijadas
├── 🔨 Makefile                            # Comandos de utilidad
│
├── 📂 sql/
│   └── 01_ddl_schema.sql                 # DDL: esquema estrella + vistas de auditoría
│
├── 📂 etl/
│   ├── pipeline.py                       # DAG: Extract → Validate → Clean → Feature
│   ├── extractor.py                      # Nodo Extract (queries SQL)
│   ├── cleaner.py                        # Nodo Clean (imputación, invariantes)
│   └── feature_engineer.py              # Nodo Feature (RFM, Intensity, Trend, Diversity)
│
├── 📂 eda/
│   └── data_quality_report.py           # Fase III: dup_rate, orphan_rate, MCAR/MAR/MNAR, Stop/Go
│
├── 📂 ml/
│   ├── train.py                          # Entrenamiento XGBoost + SMOTE + umbral óptimo
│   ├── evaluate.py                       # Evaluación: Bootstrap CI, Lift@K, baselines
│   ├── baselines.py                      # Baselines honestos (RFM, Recency)
│   └── predict.py                        # Predicción batch/online
│
├── 📂 api/
│   ├── app.py                            # FastAPI: lifespan singleton, endpoints
│   ├── schemas.py                        # Pydantic models
│   └── predictor.py                      # Lógica de predicción + tier-based actions
│
├── 📂 tests/
│   ├── test_invariants.py               # Tests de invariantes de negocio
│   └── test_temporal_integrity.py       # Tests de no-leakage temporal
│
├── 📂 deliverables/
│   ├── contrato_retencion.pdf           # Fase I: Analytics Contract (APA)
│   ├── contrato_retencion.tex           # Fuente LaTeX del contrato
│   ├── informe_tecnico.pdf              # Informe técnico completo (APA, 24 págs.)
│   ├── informe_tecnico.tex              # Fuente LaTeX del informe
│   ├── resumen_ejecutivo.pdf            # Executive summary
│   └── references.bib                   # Bibliografía BibTeX
│
└── 📂 artifacts/                         # Artefactos generados (gitignore: data/models)
    ├── manifest.json                     # Registro de linaje de artefactos
    └── figures/                          # Figuras EDA generadas
```

---

## 🚀 Inicio Rápido

### 1. Clonar e instalar

```bash
git clone https://github.com/juanmanuelpriet/Proyecto-ciencia-ade-datos-2.git
cd "Proyecto-ciencia-ade-datos-2"
pip install -r requirements.txt
```

### 2. Ejecutar el pipeline completo

```bash
python main.py
```

Esto ejecuta en orden:
1. **Seed** — Genera datos sintéticos (5,000 clientes, ~18 meses de historial)
2. **ETL** — Extract → Validate → Clean → Feature Engineering
3. **EDA** — Reporte de calidad de datos (figuras en `artifacts/figures/`)
4. **Train** — Entrena XGBoost con SMOTE y ajuste dinámico de umbral
5. **Evaluate** — Métricas completas con Bootstrap CI
6. **Manifest** — Registra todos los artefactos con hash MD5

### 3. Iniciar la API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# Documentación: http://localhost:8000/docs
```

### 4. Comandos útiles

```bash
python main.py --skip-seed     # Reusar datos ya generados (más rápido)
python main.py --skip-train    # Solo regenerar EDA y figuras
python main.py --force-seed    # Forzar regeneración de datos
pytest tests/                  # Ejecutar suite de pruebas
make help                      # Ver todos los comandos disponibles
```

---

## 📊 Resultados

| Métrica | Baseline RFM | Baseline Recency | **Modelo XGBoost** |
|---------|-------------|-----------------|-------------------|
| Recall (Churn) | 0.45 | 0.38 | **0.82 [0.80, 0.84]** |
| Precision | 0.11 | 0.09 | **0.45** |
| AUC-ROC | 0.62 | 0.58 | **0.87** |
| AUC-PR | 0.14 | 0.11 | **0.52** |
| Lift@20% | 1.20 | 1.10 | **2.40x** |
| Umbral τ* | 0.50 (fijo) | 60d (fijo) | **0.31 (optimizado)** |

> Los valores en corchetes son Intervalos de Confianza del 95% (Bootstrap, B=1000).

---

## 🔬 Metodología

### Fases del Proyecto

| Fase | Descripción | Artefacto Principal |
|------|-------------|---------------------|
| **I** | Definición y Analytics Contract | `deliverables/contrato_retencion.pdf` |
| **II** | Diseño de Datos y Modelo Temporal | `sql/01_ddl_schema.sql`, `config.yaml` |
| **III** | Calidad de Datos y EDA | `eda/data_quality_report.py`, `artifacts/figures/` |
| **IV** | Arquitectura y Reproducibilidad | `main.py`, `artifacts/manifest.json` |
| **V** | Baselines y Evaluación | `ml/baselines.py`, `ml/evaluate.py` |
| **VI** | Feature Engineering y Evidencia | `etl/feature_engineer.py`, `deliverables/informe_tecnico.pdf` |

### Prevención de Data Leakage

Todas las features se calculan con datos estrictamente anteriores a `t₀ = 2023-09-30`. Las etiquetas de Churn se definen exclusivamente en la ventana `(t₀, t₀ + 30 días]`. El Time-based split garantiza que el conjunto de test no tenga solapamiento temporal con el de entrenamiento.

### Familia de Features

| Familia | Features | Descripción |
|---------|----------|-------------|
| **Recency** | `recency_days`, `days_since_registration` | Tiempo desde última interacción |
| **Frequency** | `frequency_orders`, `monetary_total`, `monetary_avg_order` | Volumen de compras |
| **Intensity** | `spending_intensity`, `support_tickets_90d`, `avg_delivery_days`, `return_rate` | Intensidad y fricción |
| **Trend** | `trend_slope_30d` | Deterioro del comportamiento |
| **Diversity** | `category_diversity`, `preferred_category`, `customer_segment` | Amplitud de compra |

### Manejo de Desbalanceo (~10% Churn)

- **SMOTE** (Chawla et al., 2002): sobremuestreo sintético solo en el conjunto de entrenamiento
- **`scale_pos_weight`** dinámico: backup calculado como `n_neg / n_pos`
- **Optimización de umbral**: grid-search en [0.20, 0.60] maximizando Recall sujeto a Precision ≥ 0.30

---

## 📐 Configuración (config.yaml)

```yaml
project:
  name: "Global-E-Shop Churn Predictor"
  version: "1.0.0"
  random_seed: 245573          # Semilla global → reproducibilidad determinística

temporal:
  cutoff_date: "2023-09-30"
  observation_window_days: 90
  prediction_window_days: 30

model:
  classifier: "XGBoostClassifier"
  hyperparams:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6

evaluation:
  bootstrap_iterations: 1000
  bootstrap_alpha: 0.05
```

---

## 🗄️ Esquema de Base de Datos

```
dim_customers ──── fact_orders ──── fact_order_items
      │                                    │
      └──── fact_support_tickets       dim_products
      │
      ├──── ml_churn_labels        (etiquetas temporales)
      ├──── ml_feature_store       (features point-in-time)
      └──── ml_predictions_log     (audit trail en producción)

Vistas de Auditoría:
  v_key_integrity       → dup_rate, orphan_rate por tabla
  v_explosion_factors   → explosion(A,B) para joins críticos
  v_business_invariants → violaciones de invariantes temporales
  v_feature_coverage    → cobertura del Feature Store
  v_stop_go_summary     → diagnóstico PASS/FAIL de la Regla Stop/Go
```

---

## 🔁 Reproducibilidad

Este proyecto alcanza **Reproducibilidad a Nivel de Proyecto**: ejecutar el pipeline con los mismos parámetros y datos produce artefactos con hash MD5 idénticos.

```bash
# Verificar reproducibilidad
python main.py
cat artifacts/manifest.json | python -m json.tool | grep md5
```

**Garantías:**
- Semilla global fija: `random_seed: 245573`
- Dependencias fijadas en `requirements.txt`
- Datos sintéticos reproducibles (misma semilla → mismos datos)
- Modelo XGBoost determinístico con semilla fija
- Todos los artefactos registrados en `artifacts/manifest.json` con hash MD5 y git commit

---

## 📦 Dependencias Principales

```
xgboost>=1.7.0          # Modelo de clasificación
imbalanced-learn>=0.10  # SMOTE para desbalanceo
pandas>=1.5             # Manipulación de datos
scikit-learn>=1.1       # Pipeline ML y métricas
fastapi>=0.100          # API de serving
pydantic>=2.0           # Validación de esquemas
loguru>=0.7             # Logging estructurado
pyyaml>=6.0             # Configuración
matplotlib>=3.6         # Visualizaciones
seaborn>=0.12           # Heatmaps y distribuciones
```

---

## 📄 Entregables

Los tres entregables obligatorios del proyecto están en `deliverables/`:

| Entregable | Archivo | Descripción |
|-----------|---------|-------------|
| **Repositorio** | Este repo | Código + datos + configuración |
| **Informe Técnico** | `informe_tecnico.pdf` | Documento APA completo (24 págs.) con Analytics Contract, Data Quality Report, Feature Spec y Evidence Bundle Index |
| **Reproducibilidad** | `informe_tecnico.pdf` (Apéndice) + `main.py` | Protocolo de replicación con parámetros θ y Pipeline Stages |

---

## 👥 Autores

**Juan Manuel Prieto Corredor**  
Facultad de Ingeniería, Universidad de La Sabana  
📧 juanmanuelprietocorredor@gmail.com

**Slendy Marieth Grisales Rueda**  
Facultad de Ingeniería, Universidad de La Sabana

---

## 📚 Referencias Clave

- Chawla et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD'16*, 785–794.
- Kaufman et al. (2012). Leakage in data mining. *ACM TKDD*, 6(4), 1–21.
- Sculley et al. (2015). Hidden technical debt in ML systems. *NeurIPS*, 28, 2503–2511.
- Reichheld & Schefter (2000). E-loyalty. *Harvard Business Review*, 78(4), 105–113.
