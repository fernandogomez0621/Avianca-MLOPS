# Plantilla ML — BPT · 

Estructura completa para un producto ML sobre Databricks + Unity Catalog.
Usa esta plantilla cuando crees un modelo sin el workflow de NEO o cuando migres un modelo existente.

---

## Paso 1 — Renombrar el proyecto

Ejecuta el script de inicialización. Reemplaza los 5 valores:

```powershell
.\init.ps1 `
  -ProductName   "demand_forecast_model" `
  -Domain        "ops" `
  -AreaNegocio   "scm" `
  -Proyecto      "demand_forecasting" `
  -ServicePrincipal "sp-<dominio>-prd@<organizacion>" `
  -ServiceConnection "sc-ops-databricks"
```

El script renombra la carpeta `src\flight_delay_model\`, reemplaza todos los placeholders
en los archivos YAML y Python, y muestra un resumen de qué cambió.

---

## Paso 2 — Completar los TODOs de negocio

Busca los comentarios `# TODO:` en el proyecto:

```powershell
Select-String -Path "src\**\*.py" -Pattern "# TODO:" -Recurse
```

Los TODOs más importantes:

| Archivo | Qué completar |
|---|---|
| `src\flight_delay_model\training\train.py` | `FEATURE_COLS`, `TARGET_COL`, lógica del pipeline |
| `src\flight_delay_model\evaluation\thresholds.py` | Umbrales de métricas y slices de negocio |
| `src\flight_delay_model\data_prep\notebooks\DataPreparation.py` | Tablas fuente Silver/Gold |
| `databricks.yml` | URLs reales de workspaces |

---

## Paso 3 — Configurar Azure DevOps

1. Crear (o confirmar que existe) el ARM Service Connection en Azure DevOps y verificar que el
   nombre coincide con el valor pasado a `-ServiceConnection` en el Paso 1 (ya reemplazado en
   los `serviceConnection:` de `azure-pipelines/*.yml`).

2. Crear los 3 variable groups en Azure DevOps:
   - `flight_delay_model_DASH-vg-ci` → variables: `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (QA workspace)
   - `flight_delay_model_DASH-vg-staging` → variables: `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (QA workspace)
   - `flight_delay_model_DASH-vg-prd` → variables: `DATABRICKS_HOST`, `DATABRICKS_TOKEN` (PRD workspace)

3. Crear el entorno de aprobación: `prd-approval-ops` → agregar aprobadores

4. Registrar los pipelines en Azure DevOps:
   - New Pipeline → Azure Repos Git → `azure-pipelines/ci-pipeline.yml`
   - New Pipeline → Azure Repos Git → `azure-pipelines/cd-training-pipeline.yml`
   - New Pipeline → Azure Repos Git → `azure-pipelines/cd-scoring-pipeline.yml`

5. Otorgar al Service Principal detrás del Service Connection el rol **"Service Principal User"**
   (`CAN_USE`) sobre el SP indicado en `-ServicePrincipal` (el `run_as` del target `prod`), en
   Databricks workspace prod → Identity and access → Service principals.

---

## Paso 4 — Secuencia de primer despliegue

El monitor de calidad **no se puede activar en el primer deploy** porque necesita que la tabla
de inferencia exista. Esa tabla la escribe el batch scoring job en su primera ejecución.
El bloque `quality_monitors` en `databricks.yml` (targets `qa` y `prod`) ya viene comentado con
markers `# [CD-MONITOR-QA]` / `# [CD-MONITOR-PRD]` — **no los descomentes a mano**: el script
`scripts/pipeline/enable_monitor.py` los activa automáticamente (lo ejecuta `cd-scoring-pipeline`
después de la primera corrida exitosa del batch scoring job).

> Los workflows `model-training-workflow.yml`, `batch-inference-workflow.yml` y
> `feature-materialization-workflow.yml` se incluyen desde el primer deploy — no necesitan comentarse.

**Comandos:**

```bash
databricks bundle validate --target dev
databricks bundle deploy --target dev

# Correr training (registra el modelo en UC):
databricks bundle run flight_delay_model_training_job --target dev

# Correr scoring (crea la tabla de inferencia {model_name}_inference):
databricks bundle run flight_delay_model_batch_scoring_job --target dev
```

**Resultado esperado:** tabla `gold_adv_dev.<schema>.<model_name>_inference` creada con
columnas `model_uri`, `prediction`, `scored_at` (ajustar nombres si el job usa otros).

---

### FASE 2 — Activar el monitor de calidad

**Prerequisito:** tabla `{model_name}_inference` existe en Unity Catalog (Fase 1 completada).

**Activación automática (recomendado, vía pipeline):**

`cd-scoring-pipeline` corre `enable_monitor.py databricks.yml --target qa` (y `--target prod`
en su stage correspondiente) tras la primera ejecución exitosa del batch scoring job. Esto quita
el prefijo `# [CD-MONITOR-QA]` / `# [CD-MONITOR-PRD]` de las líneas del bloque `quality_monitors`
y vuelve a desplegar el bundle con el monitor activo.

**Activación manual (para pruebas locales fuera de DevOps):**

```bash
python scripts/pipeline/enable_monitor.py databricks.yml --target qa
databricks bundle deploy --target qa
```

**Resultado esperado:** Quality Monitor creado sobre `gold_adv_qa.<schema>.<model_name>_inference`.
Verificar en Databricks UI → Data Engineering → Quality Monitoring.

---

## Paso 5 — Validar el bundle

```bash
databricks bundle validate --target dev
```

---

## Convenciones de naming BPT

| Elemento | Patrón | Ejemplo |
|---|---|---|
| Catálogo ML dev | `gold_{dominio}_dev` | `gold_ops_dev` |
| Catálogo ML qa | `gold_{dominio}_qa` | `gold_ops_qa` |
| Catálogo ML prd | `gold_{dominio}_prd` | `gold_ops_prd` |
| Experimento MLflow | `/ml/{dominio}/{area}/{proyecto}` | `/ml/ops/scm/demand_forecasting` |
| Modelo UC | `md_{area}_{subject}` | `md_scm_demand` |
| Bundle name | snake_case | `demand_forecast_model` |
