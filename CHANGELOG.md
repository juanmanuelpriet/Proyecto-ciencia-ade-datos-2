# CHANGELOG — Global-E-Shop Churn Predictor

> Historial de cambios para el proyecto de Ciencia de Datos II, Universidad de La Sabana, 2026.1

---

## [2.0.0] — 2026-04-13 · Segunda sesión de mejoras

### Nuevos archivos

#### `ml/baselines.py` — **NUEVO**
Módulo independiente de baselines honestos (antes solo existía una función mínima en `ml/evaluate.py`).

**Contiene:**
- `baseline_rfm_heuristic(X, recency_percentile=0.75)` — Umbral en percentil 75 de recency
- `baseline_recency_fixed(X, threshold_days=60)` — Regla fija: sin compra en >60 días
- `baseline_low_frequency(X, min_orders=2)` — Clientes con <2 órdenes en 90 días
- `baseline_rfm_composite(X, ...)` — Score ponderado R=0.5, F=0.3, M=0.2
- `evaluate_all_baselines(X_test, y_test)` — Evaluador unificado con métricas (Recall, Precision, F1, AUC-ROC)

**Para ejecutar:** `python3 ml/baselines.py` o `make baselines`

---

#### `tests/test_invariants.py` — **COMPLETAMENTE REESCRITO**

Expandido de 5 tests básicos a 6 grupos con 20+ tests cubriendo todas las invariantes del Analytics Contract.

| Grupo | Clase | Tests |
|-------|-------|-------|
| 1 | `TestDatabaseExists` | DB existe, tablas presentes, vistas de auditoría presentes |
| 2 | `TestKeyIntegrity` | dup_rate=0% para dim_customers, fact_orders, tickets, feature_store, churn_labels |
| 3 | `TestReferentialIntegrity` | orphan_rate=0% para órdenes, tickets, items |
| 4 | `TestBusinessInvariants` | delivery≥order, resolved≥created, amounts≥0, score∈[1,5], churn∈{0,1} |
| 5 | `TestFeatureMatrix` | shape≥100, churn_label presente, churn_rate∈[1%,50%], nulos<umbral, no duplicados, features completas |
| 6 | `TestStopGo` | v_stop_go_summary todo PASS, cobertura Feature Store=100% |

**Para ejecutar:** `pytest tests/test_invariants.py -v` o `make test`

---

#### `README.md` — **NUEVO**

README profesional para GitHub con:
- Badges (Python, XGBoost, SQLite, FastAPI, MIT License)
- Diagrama ASCII del Pipeline DAG
- Diagrama temporal del modelo (t₀, W_obs, W_pred)
- Árbol completo de la estructura del proyecto
- Tabla de resultados comparativa (Baselines vs XGBoost)
- Guía de inicio rápido
- Sección de reproducibilidad
- Metodología por fases
- Esquema de base de datos

---

### Archivos modificados

#### `deliverables/contrato_retencion.tex` → **`contrato_retencion.pdf`** (15 págs.)
**Problema:** El archivo original era un documento legal de "Términos y Condiciones", no un Analytics Contract académico.

**Cambios:**
- Reescrito completamente como Analytics Contract académico en formato APA 7a edición
- Formalización del problema de decisión: actor (Marketing Manager), acciones A={a₀,a₁,a₂,a₃}, utilidad U(a,ω) = CLV·1[retained|ω,a] − cost(a)
- Hipótesis falsificable con 3 condiciones explícitas de rechazo
- Tabla de métricas auditables: Recall≥0.80, guardrails: Precision≥0.30, AUC-ROC≥0.65, Lift@20%≥1.50
- Modelo temporal con ecuaciones formales (t₀, W_obs, W_pred)
- Protocolo de Acceso a Datos (GDPR, pseudonimización)
- Tabla de riesgo de joins (explosion factors)
- Referencias APA completas (Kaufman 2012, Chen 2016, Chawla 2002, Sculley 2015, RGPD 2016, Reichheld 2000)
- Fix LaTeX: `\usepackage[english]{babel}` (español no disponible en el sistema), `\setlength{\headheight}{15pt}`

#### `deliverables/informe_tecnico.tex` → **`informe_tecnico.pdf`** (24 págs.)
**Problema:** El original tenía 5 secciones básicas e incompletas, sin cobertura de las Fases III-VI.

**Cambios:**
- Reescrito completamente como informe técnico APA 7a edición (24 páginas)
- Portada APA con Author Note
- Tabla de contenidos automática
- Fase III: tablas de dup_rate/orphan_rate, clasificación MCAR/MAR/MNAR, Stop/Go con 4 criterios formales
- Fase IV: ecuación del DAG, snippet JSON del manifest, garantías de reproducibilidad
- Fase V: tabla comparativa completa con Bootstrap CI [0.80, 0.84], Lift@20%=2.4x
- Fase VI: `longtable` con 13 features (fórmula + ventana + argumento anti-leakage)
- Evidence Bundle Index: claim → artifact_id → comando de reproducción
- Protocolo Bash de reproducción completo
- 10 referencias APA (Rubin 1976, Davis/Goadrich 2006, Fader 2005, Lundberg 2017, Gama 2014, etc.)
- Fix LaTeX: removido `\usepackage{apacite}` (no disponible), reemplazado con `\usepackage{url}` + referencias manuales APA

#### `eda/data_quality_report.py` — **COMPLETAMENTE REESCRITO**

**Cambios:**
- 5 módulos explícitos con docstrings referenciando fuentes APA
- `MISSINGNESS_TAXONOMY`: diccionario con clasificación MCAR/MAR/MNAR y tratamiento para 4 variables
- `check_key_integrity(conn)`: dup_rate + orphan_rate por tabla con queries SQL directas
- `compute_explosion_factors(conn)`: explosion(A,B) para 3 joins críticos
- `analyze_missingness(feat_matrix)`: análisis MCAR/MAR/MNAR por columna
- `evaluate_stop_go(...)`: 4 criterios formales con veredicto PASS/FAIL
- 5 funciones de visualización: distribución de clases, distribución de features, heatmap de correlación, barras de missingness, correlación con target

#### `sql/01_ddl_schema.sql` — **5 vistas de auditoría agregadas**

**Vistas nuevas:**
```sql
v_key_integrity       -- dup_rate y orphan_rate por tabla (para Stop/Go)
v_explosion_factors   -- explosion(A,B) para 3 joins críticos
v_business_invariants -- detección de violaciones temporales
v_feature_coverage    -- % de cobertura del Feature Store
v_stop_go_summary     -- diagnóstico unificado PASS/FAIL por criterio
```

**Uso:** `SELECT * FROM v_stop_go_summary;`

#### `Makefile` — **CORREGIDO**

**Problema:** Usaba rutas hardcodeadas de MiKTeX para macOS que no funcionan en Linux ni en otras instalaciones.

**Cambios:**
- Rutas MiKTeX (`/Applications/MiKTeX Console.app/...`) → `pdflatex` estándar del PATH
- Funciona con MacTeX (Homebrew), TeX Live (Linux/apt), MiKTeX (Windows/scoop)
- Target `report` compilado con `cd deliverables && pdflatex ...` (evita problemas de `-output-directory`)
- Verificación de `pdflatex` en PATH antes de compilar (error claro si no está instalado)
- Nuevo target `baselines` para ejecutar `ml/baselines.py` standalone
- Nuevo target `seed` para regenerar solo datos sintéticos
- Target `help` mejorado con descripción de todos los comandos
- `clean` ahora también borra archivos auxiliares LaTeX en `deliverables/` y cachés de pytest

#### `requirements.txt` — **ACTUALIZADO**

**Dependencias agregadas:**
- `shap==0.45.0` — SHAP feature importance (TreeExplainer para XGBoost, referenciado en el informe)
- `pypdf==4.2.0` — Lectura/validación de PDFs (scripts de auditoría)

#### `.gitignore` — **ACTUALIZADO**

**Cambios:**
- Agregado `*.synctex(busy)` (archivo temporal de LaTeX)
- Agregado IDEs: `.vscode/`, `.idea/`, `*.swp`, `*~`
- Agregado `VALIDACION_PROYECTO.md`, `PROMPT_MASTER.md`, `README_VALIDACION.md`, `RESUMEN_VALIDACION.docx` (documentos internos)
- `!deliverables/*.pdf` — Los PDFs compilados SÍ se versionan
- `!artifacts/manifest.json` — El manifest de linaje SÍ se versiona (fix: antes estaba en ignore)

---

## [1.0.0] — sesión inicial · Estructura base

- Creación del proyecto: `main.py`, `config.yaml`, esquema estrella SQLite
- ETL: `etl/pipeline.py`, `etl/extractor.py`, `etl/cleaner.py`, `etl/feature_engineer.py`
- ML: `ml/train.py`, `ml/evaluate.py`, `ml/predict.py`
- API: `api/app.py`, `api/schemas.py`, `api/predictor.py`
- Tests iniciales: `tests/test_temporal_integrity.py`
- SQL DDL: `sql/01_ddl_schema.sql`
- EDA básico: `eda/data_quality_report.py` (versión inicial)

---

## Guía para antigravity — Comandos para ejecutar

### Instalar dependencias actualizadas
```bash
cd "/Users/juanmanuelprieto/Documents/proyect-cien-2/proyecto 1"
pip install -r requirements.txt
```

### Ejecutar pipeline completo
```bash
python3 main.py
```

### Ejecutar solo los tests
```bash
pytest tests/ -v --tb=short
```

### Recompilar los PDFs
```bash
make report
# Requiere: pdflatex en PATH (brew install --cask mactex en macOS)
```

### Ver resumen de la base de datos (Stop/Go)
```bash
python3 -c "
import sqlite3, pandas as pd, yaml
cfg = yaml.safe_load(open('config.yaml'))
conn = sqlite3.connect(cfg['paths']['db'])
print(pd.read_sql('SELECT * FROM v_stop_go_summary', conn).to_string())
conn.close()
"
```

### Subir a GitHub (hacer el push)
```bash
cd "/Users/juanmanuelprieto/Documents/proyect-cien-2/proyecto 1"
git push origin main
# Si pide credenciales:
#   Username: juanmanuelpriet
#   Password: <tu GitHub Personal Access Token (ghp_...)>
```
