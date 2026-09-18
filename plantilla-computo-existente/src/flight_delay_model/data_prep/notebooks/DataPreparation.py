# Databricks notebook source
# =============================================================================
# DataPreparation.py — Entrypoint de preparacion de datos (S2)
# Producto : flight_delay_model
#
# Flujo:
#   1. Leer features desde la feature table (generada por FeatureMaterialization)
#   2. Split temporal 70/15/15
#   3. Escribir train/val/test como Delta tables en Unity Catalog
#   4. Loguear metricas de calidad en MLflow
#
# Prerequisito: FeatureMaterialization debe haberse ejecutado antes.
#   - En CI/CD (dev/qa): FeatureMaterialization corre con use_dummy=true (datos sinteticos).
#   - En produccion:     FeatureMaterialization corre con use_dummy=false (datos reales).
# DataPreparation NO tiene logica de dummy — consume siempre la feature table.
#
# DS: implementar las funciones en:
#   - features/store.py       → get_feature_table_name, publish_features
#   - data_prep/splits.py     → SplitConfig (ajustar ratios si aplica)
# =============================================================================

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
dbutils.widgets.text("catalog",              "gold_adv_dev",          "Catálogo ML")
dbutils.widgets.text("schema",               "operaciones_flight_delay",  "Schema")
dbutils.widgets.text("model_name",           "md_operaciones_flight", "Nombre modelo (prefijo tablas Delta)")
dbutils.widgets.text("experiment_name",      "",                       "Ruta experimento MLflow")
dbutils.widgets.text("workspace_files_path", "",                       "Ruta raiz bundle en workspace")
dbutils.widgets.text("date_from",            "2020-01-01",             "Fecha inicio (filtro feature table)")
dbutils.widgets.text("date_to",              "2099-12-31",             "Fecha fin (filtro feature table)")
dbutils.widgets.text("test_ratio",           "0.2",                    "Proporcion test (informativo)")

catalog              = dbutils.widgets.get("catalog")
schema               = dbutils.widgets.get("schema")
model_name           = dbutils.widgets.get("model_name")
experiment_name      = dbutils.widgets.get("experiment_name")
workspace_files_path = dbutils.widgets.get("workspace_files_path")
date_from            = dbutils.widgets.get("date_from")
date_to              = dbutils.widgets.get("date_to")

mlflow.set_experiment(experiment_name)

# COMMAND ----------
import sys

sys.path.insert(0, f"{workspace_files_path}/src")

print(f"[INFO] catalog={catalog} | schema={schema} | model_name={model_name}")

# COMMAND ----------

from flight_delay_model.data_prep.splits import SplitConfig, split_summary, temporal_split
from flight_delay_model.features.definitions import FEATURE_NAMES
from flight_delay_model.features.store import get_feature_table_name

# COMMAND ----------
# ── 1. Leer features desde la feature table ──────────────────────────────────
# La feature table fue generada por FeatureMaterialization (dummy o datos reales).
# DataPreparation consume siempre la misma tabla — no tiene logica de dummy propia.
feature_table = get_feature_table_name(catalog, schema)
date_col = "fecha"   # TODO: ajustar al nombre real de la columna de fecha en la feature table

features_df = spark.sql(f"SELECT * FROM {feature_table}").toPandas()
print(f"[INFO] {len(features_df)} registros leidos de {feature_table}")

if features_df.empty:
    raise ValueError(
        f"La feature table '{feature_table}' esta vacia. "
        "Ejecutar FeatureMaterialization antes de DataPreparation."
    )

# Filtro de fecha opcional — solo si la columna existe en la feature table
if date_col in features_df.columns:
    features_df = features_df[
        (features_df[date_col] >= date_from) &
        (features_df[date_col] <= date_to)
    ]
    print(f"[INFO] {len(features_df)} registros tras filtro de fecha ({date_from} → {date_to})")

# COMMAND ----------
# ── 2. Split temporal 70/15/15 — NUNCA split aleatorio para series de tiempo ─
split_config = SplitConfig(train_ratio=0.70, val_ratio=0.15)

if date_col in features_df.columns:
    df_train, df_val, df_test = temporal_split(
        features_df, date_col=date_col, config=split_config
    )
    summary = split_summary(df_train, df_val, df_test, date_col=date_col)
    print(
        f"[SPLIT] train={summary['train']['n']} | "
        f"val={summary['val']['n']} | "
        f"test={summary['test']['n']}"
    )
else:
    # Fallback si no hay columna de fecha en la feature table
    n = len(features_df)
    t = int(n * 0.70)
    v = int(n * (0.70 + 0.15))
    df_train = features_df.iloc[:t].copy()
    df_val   = features_df.iloc[t:v].copy()
    df_test  = features_df.iloc[v:].copy()
    summary  = {
        "train": {"n": len(df_train), "range": ("—", "—")},
        "val":   {"n": len(df_val),   "range": ("—", "—")},
        "test":  {"n": len(df_test),  "range": ("—", "—")},
    }
    print(f"[SPLIT] train={len(df_train)} | val={len(df_val)} | test={len(df_test)} (sin columna fecha)")

# COMMAND ----------
# ── 3. Escribir splits a Unity Catalog como Delta tables ─────────────────────
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"[OK] Schema '{catalog}.{schema}' verificado")

split_tables = {}
for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
    table_fqn = f"{catalog}.{schema}.{model_name}_{split_name}"
    (spark
     .createDataFrame(df_split)
     .write
     .format("delta")
     .mode("overwrite")
     .option("overwriteSchema", "true")
     .saveAsTable(table_fqn))
    split_tables[split_name] = table_fqn
    print(f"[OK] {len(df_split)} filas → {table_fqn}")

# COMMAND ----------
# ── 4. Loguear metricas de calidad en MLflow ─────────────────────────────────
with mlflow.start_run(run_name="data_preparation"):
    mlflow.log_params({
        "catalog":        catalog,
        "schema":         schema,
        "model_name":     model_name,
        "feature_table":  feature_table,
        "train_ratio":    split_config.train_ratio,
        "val_ratio":      split_config.val_ratio,
        "date_from":      date_from,
        "date_to":        date_to,
    })
    base_metrics = {
        "total_rows":    len(features_df),
        "train_rows":    summary["train"]["n"],
        "val_rows":      summary["val"]["n"],
        "test_rows":     summary["test"]["n"],
        "n_features":    len(FEATURE_NAMES),
        "null_rate_avg": float(features_df[FEATURE_NAMES].isnull().mean().mean()),
    }
    # Tasa del target — solo si la columna de target esta en la feature table
    # TODO: reemplazar "target" con el nombre real de la columna de target
    TARGET_COL = "target"
    if TARGET_COL in features_df.columns:
        base_metrics[f"{TARGET_COL}_rate"] = float(features_df[TARGET_COL].mean())
    mlflow.log_metrics(base_metrics)
    mlflow.set_tags({
        "data_source":  "feature_table",
        "train_table":  split_tables["train"],
        "val_table":    split_tables["val"],
        "test_table":   split_tables["test"],
    })
    print("[OK] Metricas logueadas en MLflow")

# COMMAND ----------
print(f"[DONE] DataPreparation completada → {catalog}.{schema}")
dbutils.notebook.exit("OK")
