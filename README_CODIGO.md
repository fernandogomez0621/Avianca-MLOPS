# flight_delay_model — MLOps gobernado por especificaciones

Dos variantes del mismo bundle. La diferencia es **un parametro del target**,
no un cambio de arquitectura ni de contratos.

| | `plantilla-computo-existente/` | `entorno-de-evidencia/` |
|---|---|---|
| Computo | `existing_cluster_id` por ambiente | serverless (sin campo de computo) |
| Catalogos | `gold_adv_{dev,qa,prd}` | `workspace` + schema por ambiente |
| `run_as` en PRD | service principal | usuario (sin SP disponible) |
| Proposito | entrega | ejecucion real de la evidencia |

**El codigo de `src/` es identico en ambas.** Los contratos (`definitions.py`,
`thresholds.py`), los notebooks y la logica de promocion no cambian.

## plantilla-computo-existente/ — la entregable

Asume computo ya provisionado, que es el caso corporativo habitual por costo
y por politicas de plataforma. Completar antes de desplegar:

- `<workspace-host>` en los tres targets
- `TODO_CLUSTER_ID_{DEV,QA,PRD}` y `TODO_CLUSTER_ID_MATERIALIZATION`
- `TODO_SP_PRD` (service principal de produccion)
- `<grupo-owner>` y `<grupo-consumidor>` en permissions

```bash
databricks bundle validate -t dev
databricks bundle deploy   -t dev
```

## entorno-de-evidencia/ — lo que se ejecuto

Variante usada para producir las capturas del documento. Ciclo completo
corrido en dev, qa y prd: feature table, entrenamiento con veredicto PASS,
registro con alias @champion, promocion QA->PRD via copy_model_version,
inferencia batch, monitor con metricas y agente de negocio trazado.

## Orden de ejecucion

```bash
# dev / qa
databricks bundle run flight_delay_model_feature_materialization_job -t <target>
databricks bundle run flight_delay_model_training_job                -t <target>
databricks bundle run flight_delay_model_batch_scoring_job           -t <target>
databricks bundle deploy -t <target>     # crea el monitor (requiere tabla _inference)

# prd — NO entrena: promueve el @champion validado en QA
databricks bundle run flight_delay_model_model_promotion_prd_job -t prod
databricks bundle run flight_delay_model_feature_materialization_job -t prod
databricks bundle run flight_delay_model_batch_scoring_job           -t prod
databricks bundle deploy -t prod
```

## Nota sobre el monitor

`quality_monitors` requiere que la tabla `_inference` exista. Por eso el primer
`deploy` de un ambiente nuevo falla en ese recurso y hay que repetirlo despues
del primer scoring. Es una dependencia real del ciclo, no un error de config.
