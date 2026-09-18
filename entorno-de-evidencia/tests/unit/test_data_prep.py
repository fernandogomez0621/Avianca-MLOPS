# =============================================================================
# test_data_prep.py — Tests unitarios de data_prep
# Corren sin cluster: pytest tests/unit/ -m "not compute" --tb=short
# =============================================================================

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from flight_delay_model.data_prep.splits import SplitConfig, split_summary, temporal_split
from flight_delay_model.data_prep.validation import (
    EXPECTED_SCHEMA,
    validate_nulls,
    validate_schema,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame de prueba con 100 filas y fecha."""
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(100)]
    return pd.DataFrame({
        "fecha": pd.to_datetime(dates),
        "valor": np.random.default_rng(42).random(100),
    })


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_empty_schema(sample_df):
    """Sin schema definido, no hay errores."""
    errors = validate_schema(sample_df)
    assert isinstance(errors, list)


def test_validate_schema_missing_column():
    """Una columna faltante debe reportarse como error."""
    if not EXPECTED_SCHEMA:
        pytest.skip("EXPECTED_SCHEMA vacío — definir antes de correr este test")
    first_col = next(iter(EXPECTED_SCHEMA))
    df = pd.DataFrame({"otra_col": [1, 2, 3]})
    errors = validate_schema(df)
    assert any(first_col in e for e in errors)


# ---------------------------------------------------------------------------
# validate_nulls
# ---------------------------------------------------------------------------

def test_validate_nulls_no_nulls(sample_df):
    errors = validate_nulls(sample_df)
    assert isinstance(errors, list)


def test_validate_nulls_detects_excess():
    """Columna con >MAX_NULL_RATIO nulos debe reportarse."""
    if not EXPECTED_SCHEMA:
        pytest.skip("EXPECTED_SCHEMA vacío")
    first_col = next(iter(EXPECTED_SCHEMA))
    n = 100
    df = pd.DataFrame({first_col: [None] * n})
    errors = validate_nulls(df)
    assert len(errors) > 0


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------

def test_temporal_split_sizes(sample_df):
    config = SplitConfig(train_ratio=0.7, val_ratio=0.15)
    train, val, test = temporal_split(sample_df, date_col="fecha", config=config)
    total = len(train) + len(val) + len(test)
    assert total == len(sample_df)
    assert len(train) > len(val)
    assert len(val) > 0
    assert len(test) > 0


def test_temporal_split_chronological_order(sample_df):
    """El split debe preservar el orden cronológico."""
    train, val, test = temporal_split(sample_df, date_col="fecha")
    assert train["fecha"].max() <= val["fecha"].min()
    assert val["fecha"].max() <= test["fecha"].min()


def test_temporal_split_missing_date_col(sample_df):
    with pytest.raises(ValueError, match="no encontrada"):
        temporal_split(sample_df, date_col="columna_inexistente")


def test_temporal_split_empty_df():
    with pytest.raises(ValueError, match="vacío"):
        temporal_split(pd.DataFrame({"fecha": []}), date_col="fecha")


def test_split_summary(sample_df):
    train, val, test = temporal_split(sample_df, date_col="fecha")
    summary = split_summary(train, val, test, date_col="fecha")
    assert "train" in summary and "val" in summary and "test" in summary
    assert summary["train"]["n"] + summary["val"]["n"] + summary["test"]["n"] == len(sample_df)
