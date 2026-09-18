# Databricks notebook source
# =============================================================================
# Train.py — Entrypoint de entrenamiento
# Producto : flight_delay_model | Dominio: ops | Area: operaciones
# Skill    : S5 - Training  |  
# =============================================================================

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
dbutils.widgets.removeAll()
dbutils.widgets.text("experiment_name",      "", "Experimento MLflow")
dbutils.widgets.text("workspace_files_path", "", "Ruta raiz bundle en workspace")
dbutils.widgets.dropdown("catalog",    "gold_adv_dev", ["gold_adv_dev", "gold_adv_qa"], "Catalogo ML")
dbutils.widgets.text("schema",         "operaciones_flight_delay",  "Schema UC")
dbutils.widgets.text("model_name",     "md_operaciones_flight",  "Nombre modelo UC")
dbutils.widgets.dropdown("run_mode",   "dry_run", ["disabled", "dry_run", "enabled"], "Modo")
dbutils.widgets.dropdown("enable_hpo", "false",   ["false", "true"], "Usar HPO")
dbutils.widgets.text("random_seed",    "42", "Semilla")

# COMMAND ----------
experiment_name      = dbutils.widgets.get("experiment_name")
workspace_files_path = dbutils.widgets.get("workspace_files_path")
catalog    = dbutils.widgets.get("catalog")
schema     = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
run_mode   = dbutils.widgets.get("run_mode")
enable_hpo = dbutils.widgets.get("enable_hpo").lower() == "true"
seed       = int(dbutils.widgets.get("random_seed"))

mlflow.set_experiment(experiment_name)  # configurable por target via databricks.yml

# COMMAND ----------
import sys

sys.path.insert(0, f"{workspace_files_path}/src")

if run_mode == "disabled":
    dbutils.notebook.exit("DISABLED")

if "prd" in catalog.lower():
    raise ValueError(f"[BLOCKED] Entrenamiento no permitido contra catalogo prd. catalog='{catalog}'")

env = catalog.split("_")[-1]

# COMMAND ----------
from mlflow.models import infer_signature

from flight_delay_model.training.hpo import run_hpo
from flight_delay_model.training.train import (
    DEFAULT_PARAMS,
    FEATURE_COLS,
    TARGET_COL,
    create_pipeline,
    evaluate_model,
    get_feature_importance,
    train_model,
)

# Umbrales informativos en training — S6 decide formalmente
# TODO: ajustar con los umbrales de thresholds.py
METRIC_THRESHOLDS: dict = {}

# COMMAND ----------
TRAIN_TABLE = f"{catalog}.{schema}.{model_name}_train"
VAL_TABLE   = f"{catalog}.{schema}.{model_name}_val"

train_delta_version = (
    spark.sql(f"DESCRIBE HISTORY {TRAIN_TABLE} LIMIT 1")
    .select("version").collect()[0][0]
)
train_df = spark.read.format("delta").option("versionAsOf", train_delta_version).table(TRAIN_TABLE).toPandas()
val_df   = spark.read.table(VAL_TABLE).toPandas()

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET_COL]

print(f"[DATA] train={len(train_df)} | val={len(val_df)}")

# COMMAND ----------
with mlflow.start_run(run_name="flight_delay_model_training") as run:

    if enable_hpo:
        best_params, best_metric = run_hpo(X_train, y_train, X_val, y_val)
        mlflow.log_param("hpo_enabled", True)
    else:
        best_params = DEFAULT_PARAMS.copy()
        best_params["random_state"] = seed
        mlflow.log_param("hpo_enabled", False)

    model        = train_model(create_pipeline(best_params), X_train, y_train)
    val_metrics  = evaluate_model(model, X_val, y_val)

    mlflow.log_params(best_params)
    mlflow.log_param("seed", seed)
    mlflow.log_param("feature_source_version", train_delta_version)
    mlflow.log_metrics(val_metrics)

    signature = infer_signature(X_val, model.predict(X_val))
    mlflow.sklearn.log_model(
        model, artifact_path="model",
        signature=signature, input_example=X_val.head(3),
        registered_model_name=None,  # S7 registra; S5 solo loguea
    )

    importances = get_feature_importance(model)
    if importances:
        mlflow.log_dict(importances, "feature_importance.json")

    mlflow.set_tags({
        "env":                   env,
        "candidate":             "true",
        "model_name":            model_name,
        "uc_schema":             schema,
        "feature_source_tables": f"{TRAIN_TABLE},{VAL_TABLE}",
    })

    candidate_run_id    = run.info.run_id
    candidate_model_uri = f"runs:/{candidate_run_id}/model"

    dbutils.jobs.taskValues.set(key="candidate_run_id",    value=candidate_run_id)
    dbutils.jobs.taskValues.set(key="candidate_model_uri", value=candidate_model_uri)

    print(f"[OUTPUT] candidate_run_id={candidate_run_id}")

# COMMAND ----------
dbutils.notebook.exit(candidate_run_id)
