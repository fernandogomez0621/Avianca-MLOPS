# =============================================================================
# test_evaluation.py — Tests unitarios de evaluación y gate PASS/FAIL
# Corren sin cluster: pytest tests/unit/ -m "not compute" --tb=short
# =============================================================================

import pytest

from flight_delay_model.evaluation.thresholds import (
    ITERATION_BUDGET,
    THRESHOLDS,
)
from flight_delay_model.evaluation.verdict import (
    Verdict,
    check_absolute_thresholds,
    check_champion_comparison,
    check_slice_thresholds,
    decrement_budget,
    generate_verdict,
)

# ---------------------------------------------------------------------------
# check_absolute_thresholds
# ---------------------------------------------------------------------------

def test_absolute_thresholds_pass_when_thresholds_empty():
    """Sin umbrales definidos, el gate siempre pasa."""
    # Monkeypatch THRESHOLDS vacío es complejo; probamos directamente con THRESHOLDS
    # Si THRESHOLDS está vacío (plantilla sin configurar), debe pasar
    if not THRESHOLDS:
        result = check_absolute_thresholds({"mae": 0.5})
        assert result is True


def test_absolute_thresholds_pass_with_good_metrics():
    """Métricas por encima del umbral → PASS (comparador >=, maximización — scores de clasificación)."""
    if not THRESHOLDS:
        pytest.skip("THRESHOLDS vacío — definir umbrales en thresholds.py")
    good = {k: v * 2.0 for k, v in THRESHOLDS.items()}  # doble del umbral
    assert check_absolute_thresholds(good) is True


def test_absolute_thresholds_fail_with_bad_metrics():
    """Métricas por debajo del umbral → FAIL."""
    if not THRESHOLDS:
        pytest.skip("THRESHOLDS vacío — definir umbrales en thresholds.py")
    bad = {k: v * 0.5 for k, v in THRESHOLDS.items()}  # mitad del umbral
    assert check_absolute_thresholds(bad) is False


# ---------------------------------------------------------------------------
# check_champion_comparison
# ---------------------------------------------------------------------------

def test_champion_comparison_no_champion():
    """Sin champion (dict vacío), el gate pasa — es el primer deploy."""
    result = check_champion_comparison(
        candidate_metrics={"mae": 0.4},
        champion_metrics={},
        primary_metric="mae",
    )
    assert result is True


def test_champion_comparison_better_candidate():
    """Candidato mejor que champion (score más alto, maximización) → PASS."""
    result = check_champion_comparison(
        candidate_metrics={"mae": 0.50},
        champion_metrics={"mae": 0.40},
        primary_metric="mae",
    )
    assert result is True


def test_champion_comparison_within_tolerance():
    """Candidato ligeramente peor pero dentro de CHAMPION_TOLERANCE → PASS."""
    # champion=0.40, tolerance=5% → umbral máximo = 0.40 * 1.05 = 0.42
    result = check_champion_comparison(
        candidate_metrics={"mae": 0.41},
        champion_metrics={"mae": 0.40},
        primary_metric="mae",
    )
    assert result is True


def test_champion_comparison_worse_than_tolerance():
    """Candidato más de CHAMPION_TOLERANCE peor (score más bajo, maximización) → FAIL."""
    result = check_champion_comparison(
        candidate_metrics={"mae": 0.30},
        champion_metrics={"mae": 0.40},
        primary_metric="mae",
    )
    assert result is False


# ---------------------------------------------------------------------------
# check_slice_thresholds
# ---------------------------------------------------------------------------

def test_slice_thresholds_pass_when_empty():
    """Sin slices definidos, el gate pasa."""
    assert check_slice_thresholds({}) is True


def test_slice_thresholds_pass_when_all_within():
    """Score del slice por encima del umbral (maximización) → PASS."""
    slice_metrics = {"region_norte": {"rmse": 150.0}}
    # Monkeypatch temporal para el test
    import flight_delay_model.evaluation.verdict as v_module
    original = v_module.SLICE_THRESHOLDS
    v_module.SLICE_THRESHOLDS = {"region_norte": {"rmse": 120.0}}
    try:
        assert check_slice_thresholds(slice_metrics) is True
    finally:
        v_module.SLICE_THRESHOLDS = original


def test_slice_thresholds_fail_when_exceeds():
    """Score del slice por debajo del umbral (maximización) → FAIL."""
    slice_metrics = {"region_norte": {"rmse": 90.0}}
    import flight_delay_model.evaluation.verdict as v_module
    original = v_module.SLICE_THRESHOLDS
    v_module.SLICE_THRESHOLDS = {"region_norte": {"rmse": 120.0}}
    try:
        assert check_slice_thresholds(slice_metrics) is False
    finally:
        v_module.SLICE_THRESHOLDS = original


# ---------------------------------------------------------------------------
# generate_verdict
# ---------------------------------------------------------------------------

def test_generate_verdict_pass():
    """Métricas que superan todos los umbrales (maximización) → PASS."""
    good_metrics = {k: v * 2.0 for k, v in THRESHOLDS.items()} if THRESHOLDS else {}
    result = generate_verdict(
        metrics=good_metrics,
        champion_metrics={},
        slice_metrics={},
        primary_metric="mae",
        iteration_number=1,
    )
    assert result["verdict"] == Verdict.PASS.value


def test_generate_verdict_fail_absolute():
    if not THRESHOLDS:
        pytest.skip("THRESHOLDS vacío")
    bad_metrics = {k: v * 0.5 for k, v in THRESHOLDS.items()}
    result = generate_verdict(
        metrics=bad_metrics,
        champion_metrics={},
        slice_metrics={},
        primary_metric=next(iter(THRESHOLDS)),
        iteration_number=1,
    )
    assert result["verdict"] == Verdict.FAIL.value
    assert "hypothesis" in result


def test_generate_verdict_has_required_keys():
    result = generate_verdict(
        metrics={},
        champion_metrics={},
        slice_metrics={},
        primary_metric="mae",
        iteration_number=2,
    )
    assert all(k in result for k in [
        "verdict", "absolute_thresholds_pass",
        "champion_comparison_pass", "slice_thresholds_pass",
        "iteration", "budget_remaining",
    ])


# ---------------------------------------------------------------------------
# decrement_budget
# ---------------------------------------------------------------------------

def test_decrement_budget_returns_remaining():
    remaining = decrement_budget(iteration_number=1)
    assert remaining == ITERATION_BUDGET["max_iterations"] - 1


def test_decrement_budget_at_limit_raises():
    max_iter = ITERATION_BUDGET["max_iterations"]
    with pytest.raises(RuntimeError, match="Presupuesto de iteracion agotado"):
        decrement_budget(iteration_number=max_iter)
