# =============================================================================
# test_features.py — Tests unitarios de feature engineering
# Corren sin cluster: pytest tests/unit/ -m "not compute" --tb=short
# =============================================================================

import numpy as np
import pandas as pd
import pytest

from flight_delay_model.features.definitions import (
    FEATURE_CATALOG,
    FEATURE_NAMES,
    compute_features,
)
from flight_delay_model.features.validation import (
    run_feature_validations,
    validate_feature_columns,
    validate_feature_nulls,
)

# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------

def test_compute_features_raises_when_catalog_empty():
    """Sin features definidas en FEATURE_CATALOG, debe lanzar NotImplementedError."""
    if FEATURE_CATALOG:
        pytest.skip("FEATURE_CATALOG ya definido — test de placeholder no aplica")
    with pytest.raises(NotImplementedError, match="FEATURE_CATALOG está vacío"):
        compute_features(pd.DataFrame({"col": [1, 2, 3]}))


def test_compute_features_returns_all_feature_names():
    """Si FEATURE_CATALOG está completo, compute_features devuelve todas las features."""
    if not FEATURE_CATALOG:
        pytest.skip("FEATURE_CATALOG vacío — completar definitions.py")
    source_cols = {f.source_col for f in FEATURE_CATALOG}
    df = pd.DataFrame({col: np.random.default_rng(42).random(50) for col in source_cols})
    result = compute_features(df)
    assert list(result.columns) == FEATURE_NAMES


def test_compute_features_missing_source_col():
    """Columna fuente faltante debe lanzar ValueError."""
    if not FEATURE_CATALOG:
        pytest.skip("FEATURE_CATALOG vacío")
    with pytest.raises(ValueError, match="no encontrada"):
        compute_features(pd.DataFrame({"columna_que_no_existe": [1]}))


# ---------------------------------------------------------------------------
# validate_feature_columns
# ---------------------------------------------------------------------------

def test_validate_feature_columns_empty_feature_names():
    """Sin features definidas no hay errores de columnas."""
    errors = validate_feature_columns(pd.DataFrame({"col": [1]}))
    assert isinstance(errors, list)


def test_validate_feature_columns_missing():
    if not FEATURE_NAMES:
        pytest.skip("FEATURE_NAMES vacío")
    errors = validate_feature_columns(pd.DataFrame({"columna_random": [1]}))
    assert len(errors) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# validate_feature_nulls
# ---------------------------------------------------------------------------

def test_validate_feature_nulls_no_features():
    errors = validate_feature_nulls(pd.DataFrame())
    assert errors == []


# ---------------------------------------------------------------------------
# run_feature_validations
# ---------------------------------------------------------------------------

def test_run_feature_validations_passes_with_valid_data():
    if not FEATURE_NAMES:
        pytest.skip("FEATURE_NAMES vacío")
    df = pd.DataFrame({f: np.random.default_rng(0).random(100) for f in FEATURE_NAMES})
    # No debe lanzar excepción
    run_feature_validations(df)
