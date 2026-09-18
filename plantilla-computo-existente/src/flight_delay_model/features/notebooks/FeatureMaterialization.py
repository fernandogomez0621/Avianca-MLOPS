# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# =============================================================================
# FeatureMaterialization.py — Entrypoint de feature engineering (S4)
# Producto : flight_delay_model
# Lee Silver/Gold, calcula features (definitions.py), valida y publica
# al feature store de Unity Catalog.
# =============================================================================

# COMMAND ----------
dbutils.widgets.text("catalog",              "gold_adv_dev",   "Catálogo ML destino")
dbutils.widgets.text("schema",               "adv_dev_flight_delay", "Schema destino")
dbutils.widgets.text("workspace_files_path", "",               "Ruta raiz bundle en workspace")
dbutils.widgets.text("start_date",           "",               "Fecha inicio (YYYY-MM-DD, vacío = 90 días atrás)")
dbutils.widgets.text("end_date",             "",               "Fecha fin (YYYY-MM-DD, vacío = hoy)")
dbutils.widgets.dropdown("use_dummy", "false", ["true", "false"],
                         "Usar datos dummy en lugar de tabla fuente")
dbutils.widgets.text("dummy_rows", "3000", "Registros dummy (solo si use_dummy=true)")

catalog              = dbutils.widgets.get("catalog")
schema               = dbutils.widgets.get("schema")
workspace_files_path = dbutils.widgets.get("workspace_files_path")
start_date           = dbutils.widgets.get("start_date") or None
end_date             = dbutils.widgets.get("end_date")   or None
use_dummy            = dbutils.widgets.get("use_dummy").lower() == "true"
dummy_rows           = int(dbutils.widgets.get("dummy_rows"))

import sys

sys.path.insert(0, f"{workspace_files_path}/src")

from datetime import date, timedelta

if start_date is None:
    start_date = str(date.today() - timedelta(days=90))
if end_date is None:
    end_date = str(date.today())

print(f"[INFO] catalog={catalog} | rango: {start_date} → {end_date} | use_dummy={use_dummy}")

# COMMAND ----------
from flight_delay_model.data_prep.extraction import (
    ExtractionConfig,
    build_extraction_query,
    generate_dummy_data,
)
from flight_delay_model.features.definitions import FEATURE_NAMES, compute_features
from flight_delay_model.features.store import get_feature_table_name, publish_features
from flight_delay_model.features.validation import run_feature_validations

# COMMAND ----------
# 1. Extraer datos fuente (real o dummy)
if use_dummy:
    raw_df = generate_dummy_data(n_rows=dummy_rows)
    print(f"[INFO] Datos dummy generados: {len(raw_df)} registros")
else:
    config = ExtractionConfig(
        catalog=catalog,
        start_date=start_date,
        end_date=end_date,
        source_table="",  # TODO: definir la tabla fuente real en ExtractionConfig
    )
    query = build_extraction_query(config)
    raw_df = spark.sql(query).toPandas()
    print(f"[INFO] Registros extraídos: {len(raw_df)}")

# COMMAND ----------
# 2. Calcular features (lógica compartida con BatchInference — AP6)
features_pdf = compute_features(raw_df)
print(f"[INFO] Features calculadas: {FEATURE_NAMES}")

# COMMAND ----------
# 3. Añadir columnas de control al feature store
# Estas columnas son necesarias para los consumidores de la feature table:
#   - client_id : clave primaria del feature store
#   - fecha     : columna de fecha para split temporal en DataPreparation
#   - target    : variable objetivo requerida por Train.py
# TODO: ajustar los nombres de columna al identificador y target del dominio real
for col in ["client_id", "fecha", "target"]:  # TODO: reemplazar "target" con el nombre real
    if col in raw_df.columns:
        features_pdf[col] = raw_df[col].values

# COMMAND ----------
# 4. Validar calidad de features antes de publicar (solo las columnas finales)
run_feature_validations(features_pdf[FEATURE_NAMES])
print("[OK] Validación de features pasada")

# COMMAND ----------
# 5. Publicar al feature store de Unity Catalog
features_sdf = spark.createDataFrame(features_pdf)
publish_features(spark=spark, catalog=catalog, features_df=features_sdf, schema=schema)

table_name = get_feature_table_name(catalog, schema)
print(f"[OK] {len(features_pdf)} registros publicados en {table_name} | columnas: {list(features_pdf.columns)}")

# COMMAND ----------
dbutils.notebook.exit(f"OK:{len(features_pdf)}")
