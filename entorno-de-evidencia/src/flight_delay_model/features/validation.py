# =============================================================================
# validation.py — Validaciones de features antes de publicar al feature store
# Producto : flight_delay_model
# Lógica pura — testeable con pandas.
# =============================================================================

import pandas as pd

from .definitions import FEATURE_NAMES

# TODO: definir rangos válidos para cada feature
# Formato: {nombre_feature: (min, max)}
FEATURE_VALID_RANGES: dict[str, tuple] = {
    # TODO: completar con los rangos de negocio esperados
    # Ejemplo: "dias_desde_ultima_compra": (0, 3650),
}

MAX_NULL_RATIO: float = 0.02  # máximo 2% nulos en features


def validate_feature_columns(df: pd.DataFrame) -> list[str]:
    """Verifica que el DataFrame tiene todas las features esperadas."""
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    return [f"Feature faltante: '{f}'" for f in missing]


def validate_feature_nulls(df: pd.DataFrame) -> list[str]:
    """Verifica que las features no superan el umbral de nulos."""
    errors = []
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            continue
        ratio = df[feat].isna().mean()
        if ratio > MAX_NULL_RATIO:
            errors.append(f"Nulos en '{feat}': {ratio:.1%} > {MAX_NULL_RATIO:.1%}")
    return errors


def validate_feature_ranges(df: pd.DataFrame) -> list[str]:
    """Verifica rangos válidos de las features numéricas."""
    errors = []
    for feat, (lo, hi) in FEATURE_VALID_RANGES.items():
        if feat not in df.columns:
            continue
        out = ((df[feat] < lo) | (df[feat] > hi)).sum()
        if out > 0:
            errors.append(f"'{feat}': {out} valores fuera de [{lo}, {hi}]")
    return errors


def run_feature_validations(df: pd.DataFrame) -> None:
    """Ejecuta todas las validaciones; lanza ValueError si hay errores."""
    errors = (
        validate_feature_columns(df)
        + validate_feature_nulls(df)
        + validate_feature_ranges(df)
    )
    if errors:
        raise ValueError(
            "Validación de features fallida:\n" + "\n".join(f"  - {e}" for e in errors)
        )
