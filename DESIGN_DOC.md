# Churn Prediction System: Technical Design & Decision Log

Este documento detalla las decisiones arquitectónicas y técnicas tomadas para el sistema de predicción de Churn de Global-E-Shop, siguiendo los estándares de **Staff Data Engineering** y **Lead Data Science**.

## 1. Data Foundation (SQL)
**Decisión:** Arquitectura de esquema en estrella con separación física entre Dimensiones (`dim_*`), Hechos (`fact_*`) y Almacenamiento de ML (`ml_*`).
- **Justificación:** La separación permite integridad referencial estricta en el nivel transaccional mientras se facilita la extracción masiva de features. El uso de `ml_feature_store` con `cutoff_date` garantiza la auditabilidad de las predicciones pasadas y previene el *data leakage* al versionar las features punto-a-punto en el tiempo.
- **Indices:** Se implementaron índices compuestos en `(customer_id, order_date)` para optimizar las queries de agregación temporal, que son el cuello de botella del pipeline.

## 2. Data Pipeline (ETL)
**Decisión:** DAG desacoplado en módulos `Extract`, `Validate`, `Clean` y `Feature`.
- **Prevalencia de Invariantes:** Se implementaron validaciones de integridad de negocio (e.g., `delivery_date >= order_date`) antes del modelado para filtrar ruido operativo.
- **Estrategia de Nulos:** Imputación por mediana para variables numéricas y "Unknown" para categóricas, preservando el sesgo original sin introducir varianza artificial.
- **Features de Negocio:** Se incluyeron métricas de **Intensity** (gasto diario) y **Trend** (pendiente de gasto) para capturar el deterioro del comportamiento antes del abandono total.

## 3. Machine Learning Rule
**Decisión:** Pipeline de `XGBoost` con `SMOTE` + `scale_pos_weight` y optimización de umbral.
- **Manejo de Desbalanceo:** Dado que el Churn es un evento raro (~10%), se combinó sobremuestreo sintético (SMOTE) para suavizar la frontera de decisión y pesos de clase en el booster para penalizar duramente los falsos negativos.
- **Optimización de Recall:** El sistema rechaza el umbral estándar de 0.5. Se realiza un grid-search dinámico para encontrar el umbral que maximiza el **Recall**, manteniendo un piso de **Precision** del 30% para asegurar la viabilidad económica de las campañas.
- **Validación:** El uso de **Recall@K** y curvas de **Lift** permite al negocio entender el impacto real: "Si contactamos al top 20% de clientes en riesgo, capturamos al X% de los churners totales".

## 4. Serving (FastAPI)
**Decisión:** Servidor asíncrono con patrón **Lifespan Singleton**.
- **Eficiencia:** El modelo se carga en RAM una sola vez al inicio (`lifespan`), reduciendo la latencia de inferencia de cientos de milisegundos a menos de 10ms.
- **Acciones Recomendadas:** El sistema no solo devuelve una probabilidad, sino una decisión accionable basada en el nivel de riesgo (Tier-based actions), alineando el ML con la ejecución de marketing.

## 5. Prevención de Data Leakage (Correctitud Temporal)
- **Regla de Oro:** Todas las features se calculan utilizando estrictamente datos previos al `cutoff_date` ($t \le t_0$).
- **Etiquetado:** El target de Churn se define exclusivamente en la ventana de predicción ($t > t_0$), asegurando que el modelo no "vea" el futuro durante el entrenamiento.
