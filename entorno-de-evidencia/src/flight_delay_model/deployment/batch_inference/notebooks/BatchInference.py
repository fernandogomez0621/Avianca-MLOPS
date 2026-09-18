# Databricks notebook source
# =============================================================================
# BatchInference.py — Entrypoint de scoring batch (deployment/batch_inference)
# Producto : flight_delay_model
# Carga @champion desde Unity Catalog y genera predicciones para el rango de fechas.
# Lee siempre desde la feature table (generada por FeatureMaterialization).
# =============================================================================

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
dbutils.widgets.text("catalog",              "gold_adv_dev",           "Catálogo ML")
dbutils.widgets.text("schema",               "operaciones_flight_delay",   "Schema Unity Catalog")
dbutils.widgets.text("model_name",           "md_operaciones_flight",  "Nombre modelo UC")
dbutils.widgets.text("workspace_files_path", "",                         "Ruta raiz bundle en workspace")
dbutils.widgets.text("start_date",           "",                         "Fecha inicio scoring (YYYY-MM-DD, vacío = ayer)")
dbutils.widgets.text("end_date",             "",                         "Fecha fin scoring (YYYY-MM-DD, vacío = hoy)")
dbutils.widgets.text("date_col",             "fecha",                    "Nombre columna de fecha en feature table")  # TODO: ajustar al nombre real

catalog              = dbutils.widgets.get("catalog")
schema               = dbutils.widgets.get("schema")
model_name           = dbutils.widgets.get("model_name")
workspace_files_path = dbutils.widgets.get("workspace_files_path")
start_date           = dbutils.widgets.get("start_date") or None
end_date             = dbutils.widgets.get("end_date")   or None
date_col             = dbutils.widgets.get("date_col")

import sys

sys.path.insert(0, f"{workspace_files_path}/src")

env = catalog.split("_")[-1]
print(f"[INFO] env={env} | catalog={catalog} | model={model_name}")

# COMMAND ----------
from datetime import date, timedelta

if start_date is None:
    start_date = str(date.today() - timedelta(days=1))
if end_date is None:
    end_date = str(date.today())

print(f"[INFO] Rango de scoring: {start_date} → {end_date}")

# COMMAND ----------
# Cargar @champion desde Unity Catalog
model_fqn  = f"{catalog}.{schema}.{model_name}"
model_uri  = f"models:/{model_fqn}@champion"

try:
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"[OK] Modelo cargado: {model_uri}")
except Exception as e:
    print(f"[ERROR] No se pudo cargar el modelo: {e}")
    raise

# COMMAND ----------
from flight_delay_model.features.definitions import FEATURE_NAMES
from flight_delay_model.features.store import get_feature_table_name

# Leer siempre desde la feature table (generada por FeatureMaterialization).
# En dev/qa: FeatureMaterialization corre con use_dummy=true (datos sinteticos).
# En prod:   FeatureMaterialization corre con use_dummy=false (datos reales).
feature_table = get_feature_table_name(catalog, schema)
features_df = spark.sql(f"""
    SELECT *
    FROM {feature_table}
    WHERE {date_col} BETWEEN '{start_date}' AND '{end_date}'
""").toPandas()
print(f"[INFO] Registros leídos de {feature_table}: {len(features_df)}")

if features_df.empty:
    print("[WARN] No hay registros para el rango de fechas. Saliendo.")
    dbutils.notebook.exit("NO_DATA")

# COMMAND ----------
# Generar predicciones — solo columnas de features (AP6)
X = features_df[FEATURE_NAMES]
predictions = model.predict(X)
features_df["prediction"] = predictions
features_df["scored_at"]  = str(date.today())
features_df["model_uri"]  = model_uri

# COMMAND ----------
# Persistir en inference table
inference_table = f"{catalog}.{schema}.{model_name}_inference"

spark.createDataFrame(features_df).write \
    .format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(inference_table)

print(f"[OK] {len(features_df)} predicciones escritas en {inference_table}")

# COMMAND ----------
dbutils.notebook.exit(f"OK:{len(features_df)}")
