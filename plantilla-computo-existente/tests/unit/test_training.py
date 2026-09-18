# =============================================================================
# test_training.py — Tests unitarios de logica pura de training
# Corren sin cluster: pytest tests/unit/ -m "not compute"
# =============================================================================

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from flight_delay_model.training.hpo import BUDGET, validate_budget
from flight_delay_model.training.train import (
    DEFAULT_PARAMS,
    FEATURE_COLS,
    TARGET_COL,
    check_thresholds,
    create_pipeline,
    evaluate_model,
    evaluate_slice,
    train_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Dataset sintetico minimo para tests unitarios.

    TODO: adaptar las columnas a FEATURE_COLS real y el tipo de TARGET_COL
    (binario/multiclase/continuo) segun el algoritmo elegido en create_pipeline()
    de train.py — rng.random(n) genera un target continuo; para clasificacion
    usar rng.integers(0, n_clases, n).
    """
    rng = np.random.default_rng(42)
    n = 200
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data[TARGET_COL] = rng.random(n)
    df = pd.DataFrame(data)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _skip_if_not_implemented(fn, *args, **kwargs):
    """Ejecuta fn; si lanza NotImplementedError, salta el test con mensaje claro."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"TODO pendiente en train.py: {e}")


def test_create_pipeline_returns_pipeline():
    pipeline = _skip_if_not_implemented(create_pipeline, DEFAULT_PARAMS)
    assert isinstance(pipeline, Pipeline)


def test_train_model_returns_fitted_pipeline(synthetic_data):
    X, y = synthetic_data
    pipeline = _skip_if_not_implemented(create_pipeline, DEFAULT_PARAMS)
    model = _skip_if_not_implemented(train_model, pipeline, X, y)
    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_evaluate_model_returns_dict(synthetic_data):
    X, y = synthetic_data
    pipeline = _skip_if_not_implemented(create_pipeline, DEFAULT_PARAMS)
    model    = _skip_if_not_implemented(train_model, pipeline, X, y)
    metrics  = _skip_if_not_implemented(evaluate_model, model, X, y)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_evaluate_slice_subsets_correctly(synthetic_data):
    """evaluate_slice filtra por una columna de segmento sin afectar las features
    del modelo — la columna de slice se agrega aparte, no reemplaza una feature
    real (el ColumnTransformer de create_pipeline ignora columnas no listadas,
    remainder="drop", así que puede convivir con X sin romper el fit)."""
    X, y = synthetic_data
    pipeline = _skip_if_not_implemented(create_pipeline, DEFAULT_PARAMS)
    model = _skip_if_not_implemented(train_model, pipeline, X, y)
    X_copy = X.copy()
    X_copy["_slice_group"] = ["A"] * 100 + ["B"] * 100
    metrics_a = evaluate_slice(model, X_copy, y, "_slice_group", "A")
    metrics_b = evaluate_slice(model, X_copy, y, "_slice_group", "B")
    assert isinstance(metrics_a, dict)
    assert isinstance(metrics_b, dict)


def test_check_thresholds_pass(synthetic_data):
    X, y    = synthetic_data
    pipeline = _skip_if_not_implemented(create_pipeline, DEFAULT_PARAMS)
    model   = _skip_if_not_implemented(train_model, pipeline, X, y)
    metrics = _skip_if_not_implemented(evaluate_model, model, X, y)
    # Umbrales muy laxos (maximización: metric >= threshold) — un umbral bajo
    # siempre debe pasar, sin importar la métrica
    generous = {k: -1.0 for k in metrics}
    assert check_thresholds(metrics, generous) is True


def test_check_thresholds_fail(synthetic_data):
    X, y    = synthetic_data
    pipeline = _skip_if_not_implemented(create_pipeline, DEFAULT_PARAMS)
    model   = _skip_if_not_implemented(train_model, pipeline, X, y)
    metrics = _skip_if_not_implemented(evaluate_model, model, X, y)
    # Umbrales imposibles (maximización: metric >= threshold) — un umbral muy
    # alto siempre debe fallar, sin importar la métrica
    strict = {k: v * 100 for k, v in metrics.items()}
    assert check_thresholds(metrics, strict) is False


def test_hpo_validate_budget_passes():
    validate_budget(BUDGET)


def test_hpo_validate_budget_missing_key():
    with pytest.raises(ValueError, match="BUDGET incompleto"):
        validate_budget({"max_trials": 10})
