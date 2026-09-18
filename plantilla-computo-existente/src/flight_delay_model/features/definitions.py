# =============================================================================
# definitions.py — Catálogo de features del proyecto
# Producto : flight_delay_model
# Lógica pura — no depende de Spark; testeable con pandas.
#
# REGLA AP6: esta misma función compute_features() se usa en training
# (Train.py) y en inference (BatchInference.py). NUNCA duplicar la lógica.
# =============================================================================

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureDefinition:
    """Metadatos de una feature."""
    name:        str
    source_col:  str          # columna original en la tabla fuente
    description: str
    dtype:       str = "float64"


REQUIRED_RAW_COLS: list[str] = [
    "client_id", "fecha", "dep_hour", "day_of_week", "month",
    "carrier", "distance_km", "prev_delay", "congestion",
]

TARGET_RAW_COL: str = "target"

FEATURE_CATALOG: list[FeatureDefinition] = [
    FeatureDefinition("dep_hour",     "dep_hour",    "Hora programada de salida (0-23)",            "int64"),
    FeatureDefinition("day_of_week",  "day_of_week", "Dia de la semana (1=lun ... 7=dom)",          "int64"),
    FeatureDefinition("month",        "month",       "Mes de operacion (1-12)",                     "int64"),
    FeatureDefinition("carrier",      "carrier",     "Operador codificado",                         "int64"),
    FeatureDefinition("distance_km",  "distance_km", "Distancia del segmento en kilometros",        "float64"),
    FeatureDefinition("prev_delay",   "prev_delay",  "Retraso en minutos del segmento anterior",    "float64"),
    FeatureDefinition("congestion",   "congestion",  "Vuelos programados en la misma hora/origen",  "int64"),
]

FEATURE_NAMES: list[str] = [f.name for f in FEATURE_CATALOG]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las features a partir del DataFrame fuente.

    Contrato AP6: misma logica en training e inference.
    Si el target esta presente se propaga (necesario para la feature table
    de entrenamiento); en inference simplemente no viene.
    """
    if not FEATURE_CATALOG:
        raise NotImplementedError("FEATURE_CATALOG esta vacio.")

    faltantes = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas fuente faltantes: {faltantes}")

    result = pd.DataFrame(index=df.index)
    for feat in FEATURE_CATALOG:
        col = df[feat.source_col]
        if feat.dtype.startswith("int"):
            result[feat.name] = col.fillna(0).astype("int64")
        else:
            result[feat.name] = col.astype("float64").fillna(col.astype("float64").median())

    cols = list(FEATURE_NAMES)
    if TARGET_RAW_COL in df.columns:
        result[TARGET_RAW_COL] = df[TARGET_RAW_COL].astype("int64")
        cols.append(TARGET_RAW_COL)

    return result[cols]
