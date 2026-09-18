# Demo — flight_delay_model (serverless)

Plantilla convertida a serverless. Placeholders resueltos:

| Variable | Valor |
|---|---|
| ProductName | flight_delay_model |
| Domain / Area / Proyecto | ops / operaciones / flight_delay |
| Modelo UC | md_operaciones_flight |
| Schema | operaciones_flight_delay |
| Catalogos | gold_adv_dev / gold_adv_qa / gold_adv_prd |

## Paso 1 — Crear catalogos y schemas (SQL Editor en Databricks)

```sql
CREATE CATALOG IF NOT EXISTS gold_adv_dev;
CREATE CATALOG IF NOT EXISTS gold_adv_qa;
CREATE CATALOG IF NOT EXISTS gold_adv_prd;

CREATE SCHEMA IF NOT EXISTS gold_adv_dev.operaciones_flight_delay;
CREATE SCHEMA IF NOT EXISTS gold_adv_qa.operaciones_flight_delay;
CREATE SCHEMA IF NOT EXISTS gold_adv_prd.operaciones_flight_delay;
```

Si `CREATE CATALOG` falla por permisos, usa el catalogo que ya exista
(p.ej. `workspace` o `main`) y cambia `catalog_name` en databricks.yml
por `main` + schemas `adv_dev`, `adv_qa`, `adv_prd`.

## Paso 2 — Validar y desplegar (desde tu maquina, NO desde el workspace)

```powershell
databricks bundle validate -t dev
databricks bundle deploy  -t dev
```

## Paso 3 — Ejecutar en orden

```powershell
databricks bundle run flight_delay_model_feature_materialization_job -t dev
databricks bundle run flight_delay_model_training_job                -t dev
databricks bundle run flight_delay_model_batch_scoring_job           -t dev
```

## Paso 4 — Evidencia para el PDF

- Catalog > gold_adv_dev > operaciones_flight_delay > Models -> alias @champion
- Experiments -> run con metricas y parametros
- Tabla `md_operaciones_flight_inference` con predicciones
- Feature table materializada

## Notas de alcance

- `run_as: service_principal_name` queda comentado en el target prod.
- CI/CD (azure-pipelines/) se entrega como especificacion + diagrama, sin ejecutar.
- ModelValidation corre en `dry_run`: registra veredicto sin bloquear.
