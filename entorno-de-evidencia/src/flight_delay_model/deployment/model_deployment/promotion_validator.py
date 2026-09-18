# =============================================================================
# promotion_validator.py — Verifica trazabilidad antes de promover a UC
# Producto : flight_delay_model
# =============================================================================

from __future__ import annotations

# Tags de trazabilidad obligatorios en el run candidato antes de promover
REQUIRED_TAGS = [
    "eval_verdict",        # debe ser "PASS"
    "candidate_run_id",    # run_id del experimento de QA
    "eval_timestamp",      # cuándo se evaluó
]


def validate_run_tags(run_id: str, client=None) -> None:
    """
    Verifica que el run MLflow tiene los tags de trazabilidad obligatorios.

    Args:
        run_id: ID del run candidato a promover.
        client: MlflowClient ya configurado. Si None, crea uno nuevo.

    Raises:
        ValueError: si falta algún tag requerido o si eval_verdict != PASS.
    """
    if client is None:
        from mlflow import MlflowClient  # lazy — no disponible fuera de Databricks
        client = MlflowClient(registry_uri="databricks-uc")

    run = client.get_run(run_id)
    tags = run.data.tags

    # Verificar que existen todos los tags requeridos
    missing = [t for t in REQUIRED_TAGS if t not in tags]
    if missing:
        raise ValueError(
            f"[BLOCKED] Faltan tags de trazabilidad en run {run_id}: {missing}. "
            "El run debe pasar por S6 (ModelValidation) antes de promover."
        )

    # Verificar que el veredicto es PASS
    verdict = tags.get("eval_verdict", "")
    if verdict != "PASS":
        raise ValueError(
            f"[BLOCKED] eval_verdict={verdict!r} para run {run_id}. "
            "Solo runs con veredicto PASS pueden promoverse a @champion."
        )
