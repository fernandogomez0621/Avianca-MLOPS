# Databricks notebook source
# =============================================================================
# ModelValidation.py — Entrypoint de evaluacion (S6)
# Producto : flight_delay_model
# =============================================================================

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
dbutils.widgets.removeAll()
dbutils.widgets.text("catalog",              "gold_adv_dev",         "Catalogo ML")
dbutils.widgets.text("schema",               "operaciones_flight_delay", "Schema UC")
dbutils.widgets.text("workspace_files_path", "",                      "Ruta raiz bundle en workspace")
dbutils.widgets.dropdown("run_mode",               "dry_run", ["disabled", "dry_run", "enabled"], "Modo")
dbutils.widgets.dropdown("enable_baseline_comparison", "true", ["true", "false"], "Comparar vs champion")

# COMMAND ----------
catalog              = dbutils.widgets.get("catalog")
schema               = dbutils.widgets.get("schema")
workspace_files_path = dbutils.widgets.get("workspace_files_path")
run_mode             = dbutils.widgets.get("run_mode")
enable_baseline      = dbutils.widgets.get("enable_baseline_comparison").lower() == "true"

import sys

sys.path.insert(0, f"{workspace_files_path}/src")

if run_mode == "disabled":
    dbutils.notebook.exit("DISABLED")

# Acoplamiento critico S5→S6: "Train" debe coincidir con task_key en model-training-workflow.yml
candidate_run_id    = dbutils.jobs.taskValues.get(taskKey="Train", key="candidate_run_id",    debugValue="REPLACE_WITH_RUN_ID")
candidate_model_uri = dbutils.jobs.taskValues.get(taskKey="Train", key="candidate_model_uri", debugValue=f"runs:/{candidate_run_id}/model")

# COMMAND ----------
import json
import os
from datetime import UTC, datetime

from flight_delay_model.evaluation.thresholds import THRESHOLDS
from flight_delay_model.evaluation.verdict import decrement_budget, generate_verdict

# TODO: cargar metricas del run candidato desde MLflow
client = mlflow.MlflowClient()
run    = client.get_run(candidate_run_id)
candidate_metrics = run.data.metrics

# TODO: cargar metricas del champion si existe
champion_metrics: dict = {}
# model_name = f"{catalog}.{schema}.md_operaciones_flight"
# try:
#     champion_mv = client.get_model_version_by_alias(model_name, "champion")
#     champion_run = client.get_run(champion_mv.run_id)
#     champion_metrics = champion_run.data.metrics
# except Exception:
#     pass

# TODO: calcular slice_metrics sobre el holdout
slice_metrics: dict = {}

# COMMAND ----------
# TODO: definir la metrica primaria (debe ser una clave de THRESHOLDS)
# Ejemplos clasificacion: "auc", "f1_score", "recall"
# Ejemplos regresion    : "rmse", "mae"
PRIMARY_METRIC   = "auc"    # TODO: ajustar al tipo de problema
iteration_number = 1        # incrementar en cada iteracion del inner loop

verdict_result = generate_verdict(
    metrics=candidate_metrics,
    champion_metrics=champion_metrics,
    slice_metrics=slice_metrics,
    primary_metric=PRIMARY_METRIC,
    iteration_number=iteration_number,
)

# Emitir veredicto como tags de trazabilidad para S7 (promotion_validator los exige)
eval_timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
with mlflow.start_run(run_id=candidate_run_id):
    mlflow.set_tag("eval_verdict",        verdict_result["verdict"])
    mlflow.set_tag("eval_primary_metric", PRIMARY_METRIC)       # trazabilidad S6→S7
    mlflow.set_tag("candidate_run_id",    candidate_run_id)     # trazabilidad S6→S7
    mlflow.set_tag("eval_timestamp",      eval_timestamp)        # trazabilidad S6→S7

dbutils.jobs.taskValues.set(key="eval_verdict", value=verdict_result["verdict"])

# Persistir evidencia para la gate NEO
evidence_path = "openspec/changes/current/evidence/ml/eval.json"
os.makedirs(os.path.dirname(evidence_path), exist_ok=True)
with open(evidence_path, "w") as f:
    json.dump({
        "candidate":  {"run_id": candidate_run_id, "model_uri": candidate_model_uri},
        "thresholds": THRESHOLDS,
        **verdict_result,
    }, f, indent=2)

print(f"[VERDICT] {verdict_result['verdict']} | run={candidate_run_id}")

if verdict_result["verdict"] == "FAIL":
    remaining = decrement_budget(iteration_number)
    print(f"[FAIL] hypothesis={verdict_result.get('hypothesis')} | budget_remaining={remaining}")
    if run_mode == "enabled":
        raise ValueError(f"Evaluacion FAIL — hipotesis: {verdict_result.get('hypothesis')}")
    else:
        print(f"[DRY_RUN] Veredicto FAIL registrado — pipeline no bloqueado (run_mode={run_mode})")

# COMMAND ----------
dbutils.notebook.exit(verdict_result["verdict"])
