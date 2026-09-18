# =============================================================================
# validation.py — Validaciones de schema y calidad de datos (data_prep)
# Producto : flight_delay_model
# Lógica pura — no depende de Spark ni Databricks; testeable con pandas.
# =============================================================================

import pandas as pd

# TODO: definir el schema esperado de los datos fuente
# Formato: {nombre_columna: dtype_esperado}
# Ejemplo: {"fecha": "datetime64[ns]", "monto": "float64", "categoria": "object"}
EXPECTED_SCHEMA: dict[str, str] = {
    # TODO: completar con las columnas reales del dataset
}

# TODO: definir rangos válidos para columnas numéricas
# Formato: {nombre_columna: (min, max)}
VALID_RANGES: dict[str, tuple] = {
    # TODO: completar con rangos de negocio
    # Ejemplo: {"monto": (0, 1_000_000), "edad": (18, 120)}
}

# Porcentaje máximo de nulos permitido por columna (0–1)
MAX_NULL_RATIO: float = 0.05


def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Verifica que el DataFrame tiene las columnas del schema esperado.

    Returns:
        Lista de mensajes de error. Vacía si el schema es correcto.
    """
    errors = []
    for col, expected_dtype in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            errors.append(f"Columna faltante: '{col}'")
        else:
            # Comparación flexible: int32/int64 y float32/float64 se consideran compatibles
            base_expected = expected_dtype.replace("64", "").replace("32", "")
            base_actual   = str(df[col].dtype).replace("64", "").replace("32", "")
            if base_actual != base_expected:
                errors.append(
                    f"Tipo incorrecto en '{col}': esperado {expected_dtype}, "
                    f"encontrado {df[col].dtype}"
                )
    return errors


def validate_nulls(df: pd.DataFrame) -> list[str]:
    """
    Verifica que las columnas requeridas no superan el umbral de nulos.

    Returns:
        Lista de mensajes de error. Vacía si los nulos están dentro del umbral.
    """
    errors = []
    for col in EXPECTED_SCHEMA:
        if col not in df.columns:
            continue
        null_ratio = df[col].isna().mean()
        if null_ratio > MAX_NULL_RATIO:
            errors.append(
                f"Nulos excesivos en '{col}': {null_ratio:.1%} > {MAX_NULL_RATIO:.1%}"
            )
    return errors


def validate_ranges(df: pd.DataFrame) -> list[str]:
    """
    Verifica que las columnas numéricas están dentro de los rangos válidos.

    Returns:
        Lista de mensajes de error. Vacía si todos los rangos son correctos.
    """
    errors = []
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
        if out_of_range > 0:
            errors.append(
                f"Valores fuera de rango en '{col}': {out_of_range} registros "
                f"fuera de [{lo}, {hi}]"
            )
    return errors


def run_all_validations(df: pd.DataFrame) -> None:
    """
    Ejecuta todas las validaciones y lanza ValueError si hay errores.

    Args:
        df: DataFrame a validar.

    Raises:
        ValueError: con todos los errores encontrados concatenados.
    """
    errors = (
        validate_schema(df)
        + validate_nulls(df)
        + validate_ranges(df)
    )
    if errors:
        raise ValueError("Validación de datos fallida:\n" + "\n".join(f"  - {e}" for e in errors))
