# Databricks notebook source
# =============================================================================
# notebook_promote.py — Entrypoint de promocion a UC (S7)
# Producto : flight_delay_model
# El modelo se registra en gold_adv_prd desde el run de QA
# =============================================================================

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
dbutils.widgets.text("catalog",              "gold_adv_dev",          "Catalogo ML destino")
dbutils.widgets.text("schema",               "operaciones_flight_delay", "Schema UC")
dbutils.widgets.text("model_name",           "md_operaciones_flight","Nombre modelo UC")
dbutils.widgets.text("workspace_files_path", "",                      "Ruta raiz bundle en workspace")
dbutils.widgets.dropdown("register_only", "false", ["true", "false"], "Solo registrar (sin promover)")

catalog              = dbutils.widgets.get("catalog")
schema               = dbutils.widgets.get("schema")
model_name           = dbutils.widgets.get("model_name")
workspace_files_path = dbutils.widgets.get("workspace_files_path")
register_only        = dbutils.widgets.get("register_only").lower() == "true"

import sys

sys.path.insert(0, f"{workspace_files_path}/src")

# S6 pasa el veredicto — S7 solo actua si es PASS
eval_verdict      = dbutils.jobs.taskValues.get(taskKey="ModelValidation", key="eval_verdict",      debugValue="PASS")
candidate_run_id  = dbutils.jobs.taskValues.get(taskKey="Train",           key="candidate_run_id",  debugValue="REPLACE_WITH_RUN_ID")

if eval_verdict != "PASS":
    print(f"[BLOCKED] eval_verdict={eval_verdict} — no se registra el modelo.")
    dbutils.notebook.exit("BLOCKED_BY_EVAL")

# COMMAND ----------
from mlflow import MlflowClient

from flight_delay_model.deployment.model_deployment.promotion_validator import (
    validate_run_tags,
)

# Verificar tags de trazabilidad antes de promover
client_check = MlflowClient(registry_uri="databricks-uc")
validate_run_tags(run_id=candidate_run_id, client=client_check)
print(f"[OK] Tags de trazabilidad verificados para run {candidate_run_id}")

client     = MlflowClient(registry_uri="databricks-uc")
model_fqn  = f"{catalog}.{schema}.{model_name}"
source_uri = f"runs:/{candidate_run_id}/model"

# Garantizar que el schema exista en el catalogo destino (requerido en primer deploy a PRD)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"[OK] Schema '{catalog}.{schema}' verificado")

# Registrar como @challenger
mv = mlflow.register_model(model_uri=source_uri, name=model_fqn)
print(f"[OK] Modelo registrado: {model_fqn} version={mv.version}")

if not register_only:
    # Promover @challenger a @champion si la evaluacion paso
    client.set_registered_model_alias(name=model_fqn, alias="champion", version=mv.version)
    print(f"[OK] Alias @champion asignado a version {mv.version}")

# COMMAND ----------
dbutils.notebook.exit(mv.version)
