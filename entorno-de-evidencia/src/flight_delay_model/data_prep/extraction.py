# =============================================================================
# extraction.py — Queries y filtros sobre tablas fuente Silver/Gold
# Producto : flight_delay_model
# =============================================================================

from dataclasses import dataclass

import pandas as pd


def generate_dummy_data(n_rows: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Genera datos sinteticos que imitan el esquema de operacion de vuelos.

    Columnas: REQUIRED_RAW_COLS + la columna target (delayed_15).
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    dep_hour     = rng.integers(0, 24, n_rows)
    day_of_week  = rng.integers(1, 8, n_rows)
    month        = rng.integers(1, 13, n_rows)
    carrier      = rng.integers(0, 6, n_rows)
    distance_km  = rng.uniform(200, 4000, n_rows)
    prev_delay   = np.clip(rng.normal(8, 20, n_rows), -15, 180)
    congestion   = rng.integers(1, 40, n_rows)

    # Probabilidad de retraso: sube con hora pico, retraso previo y congestion
    logit = (
        -4.0
        + 0.13  * dep_hour
        + 0.055 * prev_delay
        + 0.085 * congestion
        + 0.45  * (day_of_week >= 5)
        + 0.0004 * distance_km
    )
    prob = 1 / (1 + np.exp(-logit))
    delayed_15 = rng.binomial(1, prob)

    return pd.DataFrame({
        "client_id":    range(n_rows),
        "fecha":        pd.date_range("2024-01-01", periods=n_rows, freq="h").strftime("%Y-%m-%d"),
        "dep_hour":     dep_hour,
        "day_of_week":  day_of_week,
        "month":        month,
        "carrier":      carrier,
        "distance_km":  distance_km,
        "prev_delay":   prev_delay,
        "congestion":   congestion,
        "target":       delayed_15,
    })


@dataclass
class ExtractionConfig:
    """
    Configuración de extracción desde las tablas fuente.

    TODO: completar con los nombres reales de tablas y columnas.
    """
    catalog:    str
    start_date: str   # YYYY-MM-DD
    end_date:   str   # YYYY-MM-DD

    # TODO: definir las tablas fuente Silver/Gold
    # Ejemplo: source_table = "silver.ventas.transacciones"
    source_table: str = ""  # TODO: nombre real de la tabla fuente


def build_extraction_query(config: ExtractionConfig) -> str:
    """
    Construye la query SQL de extracción.

    TODO: adaptar la SELECT al esquema real del problema.
    La query debe devolver todas las columnas necesarias para
    features/ y las columnas de identificación (id, fecha).

    Returns:
        String SQL listo para ejecutar con spark.sql().
    """
    if not config.source_table:
        raise ValueError(
            "ExtractionConfig.source_table no está definido. "
            "Edita extraction.py con el nombre real de la tabla fuente."
        )

    return f"""
        SELECT
            *
            -- TODO: seleccionar solo las columnas necesarias
            -- TODO: agregar joins con tablas de referencia si aplica
        FROM {config.source_table}
        WHERE
            1=1
            -- TODO: agregar filtro de fecha con la columna correcta
            -- AND fecha_columna BETWEEN '{config.start_date}' AND '{config.end_date}'
            -- TODO: agregar filtros de calidad (excluir registros cancelados, etc.)
    """
