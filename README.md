# MLOps gobernado por especificaciones (SDD)

Propuesta y prueba de concepto para adoptar MLOps gobernado por
especificaciones sobre Databricks, Unity Catalog y Azure DevOps.

## Contenido

| Ruta | Descripción |
|---|---|
| `docs/Prueba_MLE_Propuesta.pdf` | Documento principal (3 páginas). Marco, contratos, promoción y riesgos. |
| `docs/Documento_Tecnico_Extendido.pdf` | Desarrollo completo con evidencia de implementación. |
| `plantilla-computo-existente/` | Plantilla entregable: cómputo provisionado, service principal, catálogos por ambiente. |
| `entorno-de-evidencia/` | Variante ejecutada para producir la evidencia del documento extendido. |
| `README_CODIGO.md` | Detalle de las dos variantes y orden de ejecución. |

## Idea central

Un lineamiento es una norma; una especificación es una norma verificable
por máquina. El marco convierte cada contrato del ciclo de vida en un
artefacto versionado que el pipeline lee, valida y hace cumplir.

Siete contratos gobiernan el ciclo: producto, datos, características,
entrenamiento, validación, despliegue y monitoreo. Ninguno se cumple por
disciplina individual; todos se verifican en ejecución.

## Ciclo implementado

- Tres ambientes separados por catálogo de Unity Catalog.
- Materialización de características con contrato explícito.
- Entrenamiento con veredicto automático contra umbrales y segmentos críticos.
- Registro con alias y promoción a producción por copia, tras aprobación.
- Entrenamiento excluido del ambiente productivo por diseño.
- Inferencia por lotes y en tiempo real, con tabla de inferencia trazable.
- Monitoreo de deriva y desempeño sobre los segmentos declarados.
- Consumo asistido por agente con herramientas acotadas y trazabilidad.

## Uso

Ver `README_CODIGO.md` para configuración y orden de ejecución.
