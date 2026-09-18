# Databricks notebook source
# =============================================================================
# notebook_promote_prd.py — Promocion del @champion de QA hacia PRD (S7-PRD)
# Producto : flight_delay_model
# =============================================================================
# Este notebook SOLO debe ejecutarse desde cd-scoring-pipeline (stage DeployPrd),
# despues de la aprobacion manual (environment prd-approval-ops) y con el
# service principal de PRD.
#
# A diferencia de notebook_promote.py (usado en el training job de QA), este
# notebook NO re-registra el modelo desde un run_id via taskValues — esos valores
# solo existen dentro del mismo job run y este notebook corre en un job/pipeline
# distinto. En su lugar, copia la version ya registrada y validada como @champion
# en el catalogo QA hacia el catalogo PRD via mlflow.MlflowClient.copy_model_version,
# preservando la trazabilidad (run_id, tags) del modelo original.
# =============================================================================

# COMMAND ----------
import mlflow

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------
dbutils.widgets.text("src_catalog",          "gold_adv_qa",           "Catalogo origen (QA)")
dbutils.widgets.text("dst_catalog",          "gold_adv_prd",          "Catalogo destino (PRD)")
dbutils.widgets.text("src_schema",          "adv_qa_flight_delay",  "Schema origen (QA)")
dbutils.widgets.text("dst_schema",          "adv_prd_flight_delay", "Schema destino (PRD)")
dbutils.widgets.text("model_name",           "md_operaciones_flight","Nombre modelo UC")
dbutils.widgets.text("workspace_files_path", "",                      "Ruta raiz bundle en workspace")

src_catalog          = dbutils.widgets.get("src_catalog")
dst_catalog          = dbutils.widgets.get("dst_catalog")
src_schema           = dbutils.widgets.get("src_schema")
dst_schema           = dbutils.widgets.get("dst_schema")
model_name           = dbutils.widgets.get("model_name")
workspace_files_path = dbutils.widgets.get("workspace_files_path")

import sys

sys.path.insert(0, f"{workspace_files_path}/src")

# COMMAND ----------
from mlflow import MlflowClient

from flight_delay_model.deployment.model_deployment.promotion_validator import (
    validate_run_tags,
)

client       = MlflowClient(registry_uri="databricks-uc")
src_model_fqn = f"{src_catalog}.{src_schema}.{model_name}"
dst_model_fqn = f"{dst_catalog}.{dst_schema}.{model_name}"

# Resolver la version @champion actual en QA — es la unica fuente de verdad para PRD.
champion_mv = client.get_model_version_by_alias(src_model_fqn, "champion")
print(f"[OK] Champion en QA: {src_model_fqn} version={champion_mv.version} run_id={champion_mv.run_id}")

# Verificar de nuevo los tags de trazabilidad del run original antes de copiar a PRD.
validate_run_tags(run_id=champion_mv.run_id, client=client)
print(f"[OK] Tags de trazabilidad verificados para run {champion_mv.run_id}")

# Garantizar que el schema exista en el catalogo PRD (requerido en primer deploy)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {dst_catalog}.{dst_schema}")
print(f"[OK] Schema '{dst_catalog}.{dst_schema}' verificado")

# COMMAND ----------
# Copiar la version validada de QA hacia PRD (preserva run_id/tags de origen)
src_model_uri = f"models:/{src_model_fqn}@champion"
new_mv = client.copy_model_version(src_model_uri=src_model_uri, dst_name=dst_model_fqn)
print(f"[OK] Modelo copiado a PRD: {dst_model_fqn} version={new_mv.version}")

client.set_registered_model_alias(name=dst_model_fqn, alias="champion", version=new_mv.version)
print(f"[OK] Alias @champion asignado en PRD a version {new_mv.version}")

# COMMAND ----------
dbutils.notebook.exit(new_mv.version)
