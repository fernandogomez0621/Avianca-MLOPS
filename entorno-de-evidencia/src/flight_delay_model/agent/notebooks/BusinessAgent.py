# Databricks notebook source
# =============================================================================
# BusinessAgent.py — Agente de negocio sobre el modelo en produccion
# Producto : flight_delay_model
#
# Expone el modelo gobernado (@champion) y su tabla de inferencia como
# herramientas consultables en lenguaje natural. El agente NO inventa datos:
# cada respuesta se deriva de una consulta real a Unity Catalog.
#
# Contrato: el agente solo lee. No entrena, no promueve, no escribe.
# =============================================================================

# COMMAND ----------
dbutils.widgets.text("catalog", "workspace",             "Catalogo")
dbutils.widgets.text("schema",  "adv_prd_flight_delay",  "Schema (ambiente a consultar)")
dbutils.widgets.text("model_name", "md_operaciones_flight", "Modelo UC")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")

INFERENCE_TABLE = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}_inference"
MODEL_FQN       = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
print(f"[INFO] Tabla de inferencia : {INFERENCE_TABLE}")
print(f"[INFO] Modelo              : {MODEL_FQN}")

# COMMAND ----------
# Descubrir endpoints LLM disponibles en el workspace
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
disponibles = [e.name for e in w.serving_endpoints.list()]
print("[INFO] Endpoints disponibles:")
for e in disponibles:
    print("   -", e)

PREFERIDOS = [
    "databricks-claude-sonnet-4",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-llama-4-maverick",
    "databricks-gpt-oss-120b",
]
LLM_ENDPOINT = next((m for m in PREFERIDOS if m in disponibles), None)
if LLM_ENDPOINT is None:
    LLM_ENDPOINT = next((m for m in disponibles if "llama" in m or "claude" in m or "gpt" in m), None)

assert LLM_ENDPOINT, "No hay endpoint LLM disponible en este workspace."
print(f"\n[OK] Usando endpoint: {LLM_ENDPOINT}")

llm = w.serving_endpoints.get_open_ai_client()

# COMMAND ----------
# Observabilidad del agente — mismo principio que el resto del ciclo:
# si no queda trazado, no es auditable.
import mlflow

mlflow.set_experiment(f"/Users/{w.current_user.me().user_name}/flight_delay_model_agent")
mlflow.openai.autolog()   # captura cada llamada al LLM: prompt, tool calls, latencia, tokens
print("[OK] Tracing MLflow activo — ver pestana Traces del experimento")

# COMMAND ----------
# -----------------------------------------------------------------------------
# HERRAMIENTAS — cada una es una consulta real sobre Unity Catalog
# -----------------------------------------------------------------------------
import json


@mlflow.trace(span_type="TOOL")
def modelo_en_produccion() -> dict:
    """Version, trazabilidad y veredicto de validacion del modelo @champion."""
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    from mlflow import MlflowClient

    c = MlflowClient(registry_uri="databricks-uc")
    mv = c.get_model_version_by_alias(MODEL_FQN, "champion")
    tags = {}
    try:
        tags = c.get_run(mv.run_id).data.tags
    except Exception:
        pass
    return {
        "modelo": MODEL_FQN,
        "version": mv.version,
        "alias": "champion",
        "run_id": mv.run_id,
        "veredicto_validacion": tags.get("eval_verdict"),
        "metrica_principal": tags.get("eval_primary_metric"),
        "ambiente_origen": tags.get("env"),
    }


@mlflow.trace(span_type="TOOL")
def resumen_predicciones(fecha_inicio: str = None, fecha_fin: str = None) -> dict:
    """Total de vuelos evaluados y cuantos se predicen con retraso."""
    filtro = ""
    if fecha_inicio and fecha_fin:
        filtro = f"WHERE scored_at BETWEEN '{fecha_inicio}' AND '{fecha_fin}'"
    df = spark.sql(f"""
        SELECT COUNT(*) AS vuelos_evaluados,
               SUM(prediction) AS retrasos_previstos,
               ROUND(AVG(prediction), 4) AS tasa_retraso_prevista,
               MIN(scored_at) AS desde, MAX(scored_at) AS hasta
        FROM {INFERENCE_TABLE} {filtro}
    """).toPandas()
    return df.to_dict(orient="records")[0]


@mlflow.trace(span_type="TOOL")
def retrasos_por_segmento(segmento: str = "carrier") -> list:
    """Tasa de retraso prevista por segmento: 'carrier', 'dep_hour' o 'day_of_week'."""
    if segmento not in ("carrier", "dep_hour", "day_of_week"):
        return [{"error": f"Segmento no valido: {segmento}"}]
    df = spark.sql(f"""
        SELECT {segmento},
               COUNT(*) AS vuelos,
               SUM(prediction) AS retrasos_previstos,
               ROUND(AVG(prediction), 4) AS tasa
        FROM {INFERENCE_TABLE}
        GROUP BY {segmento} ORDER BY tasa DESC LIMIT 15
    """).toPandas()
    return df.to_dict(orient="records")


@mlflow.trace(span_type="TOOL")
def desempeno_observado() -> dict:
    """Compara prediccion contra el resultado real registrado en la tabla."""
    df = spark.sql(f"""
        SELECT
          ROUND(AVG(CASE WHEN prediction = target THEN 1 ELSE 0 END), 4) AS accuracy,
          SUM(CASE WHEN prediction = 1 AND target = 0 THEN 1 ELSE 0 END) AS falsos_positivos,
          SUM(CASE WHEN prediction = 0 AND target = 1 THEN 1 ELSE 0 END) AS falsos_negativos,
          COUNT(*) AS n
        FROM {INFERENCE_TABLE}
    """).toPandas()
    return df.to_dict(orient="records")[0]


TOOLS_IMPL = {
    "modelo_en_produccion":  modelo_en_produccion,
    "resumen_predicciones":  resumen_predicciones,
    "retrasos_por_segmento": retrasos_por_segmento,
    "desempeno_observado":   desempeno_observado,
}

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "modelo_en_produccion",
        "description": "Version, run_id y veredicto de validacion del modelo @champion en produccion.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "resumen_predicciones",
        "description": "Total de vuelos evaluados, retrasos previstos y tasa. Permite filtrar por rango de fechas.",
        "parameters": {"type": "object", "properties": {
            "fecha_inicio": {"type": "string", "description": "YYYY-MM-DD"},
            "fecha_fin":    {"type": "string", "description": "YYYY-MM-DD"}}}}},
    {"type": "function", "function": {
        "name": "retrasos_por_segmento",
        "description": "Tasa de retraso prevista agrupada por segmento de negocio.",
        "parameters": {"type": "object", "properties": {
            "segmento": {"type": "string", "enum": ["carrier", "dep_hour", "day_of_week"]}},
            "required": ["segmento"]}}},
    {"type": "function", "function": {
        "name": "desempeno_observado",
        "description": "Accuracy, falsos positivos y falsos negativos comparando prediccion contra resultado real.",
        "parameters": {"type": "object", "properties": {}}}},
]

# COMMAND ----------
SYSTEM = f"""Eres un asistente de operaciones de vuelo. Respondes preguntas de negocio
consultando el modelo de prediccion de retrasos que esta en produccion.

Reglas estrictas:
- NUNCA inventes cifras. Toda cifra debe venir de una herramienta.
- Si no tienes una herramienta para la pregunta, dilo con claridad.
- Responde en espanol, breve y orientado a la decision operativa.
- Cuando cites un numero, indica de que ambiente viene ({SCHEMA}).
"""


@mlflow.trace(name="agente_negocio", span_type="AGENT")
def preguntar(pregunta: str, verbose: bool = True) -> str:
    mensajes = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": pregunta}]

    for _ in range(5):
        r = llm.chat.completions.create(
            model=LLM_ENDPOINT, messages=mensajes, tools=TOOLS_SPEC, temperature=0,
        )
        msg = r.choices[0].message
        if not msg.tool_calls:
            return msg.content

        mensajes.append(msg)
        for tc in msg.tool_calls:
            nombre = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"   [tool] {nombre}({args})")
            try:
                resultado = TOOLS_IMPL[nombre](**args)
            except Exception as exc:
                resultado = {"error": str(exc)}
            mensajes.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(resultado, default=str)})

    return "No se pudo resolver la consulta en el limite de iteraciones."

# COMMAND ----------
PREGUNTAS = [
    "Que modelo esta en produccion y paso la validacion?",
    "Cuantos vuelos se predicen con retraso y cual es la tasa?",
    "Que operador concentra mas retrasos previstos?",
    "Que tan confiable ha sido el modelo? Cuantos falsos positivos genera?",
]

for q in PREGUNTAS:
    print("=" * 78)
    print("PREGUNTA:", q)
    print(preguntar(q))
    print()
