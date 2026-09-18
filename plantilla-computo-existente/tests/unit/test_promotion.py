# =============================================================================
# test_promotion.py — Tests unitarios de validación de trazabilidad
# Corren sin cluster: pytest tests/unit/ -m "not compute" --tb=short
# =============================================================================

from unittest.mock import MagicMock

import pytest

from flight_delay_model.deployment.model_deployment.promotion_validator import (
    REQUIRED_TAGS,
    validate_run_tags,
)


def _mock_client(tags: dict):
    """Crea un MlflowClient mock con los tags indicados."""
    run = MagicMock()
    run.data.tags = tags
    client = MagicMock()
    client.get_run.return_value = run
    return client


# ---------------------------------------------------------------------------
# validate_run_tags
# ---------------------------------------------------------------------------

def test_validate_run_tags_passes_with_all_tags():
    tags = {
        "eval_verdict":     "PASS",
        "candidate_run_id": "abc123",
        "eval_timestamp":   "2026-08-06T12:00:00",
    }
    client = _mock_client(tags)
    # No debe lanzar excepción
    validate_run_tags(run_id="abc123", client=client)


def test_validate_run_tags_fails_missing_tag():
    tags = {
        "eval_verdict":     "PASS",
        # falta candidate_run_id y eval_timestamp
    }
    client = _mock_client(tags)
    with pytest.raises(ValueError, match="Faltan tags de trazabilidad"):
        validate_run_tags(run_id="abc123", client=client)


def test_validate_run_tags_fails_non_pass_verdict():
    tags = {t: "value" for t in REQUIRED_TAGS}
    tags["eval_verdict"] = "FAIL"
    client = _mock_client(tags)
    with pytest.raises(ValueError, match="eval_verdict"):
        validate_run_tags(run_id="abc123", client=client)


def test_validate_run_tags_fails_on_retrain_verdict():
    tags = {t: "value" for t in REQUIRED_TAGS}
    tags["eval_verdict"] = "RETRAIN"
    client = _mock_client(tags)
    with pytest.raises(ValueError, match="eval_verdict"):
        validate_run_tags(run_id="abc123", client=client)
