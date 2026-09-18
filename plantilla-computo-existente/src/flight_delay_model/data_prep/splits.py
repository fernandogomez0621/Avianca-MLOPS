# =============================================================================
# splits.py — Lógica de split temporal train/val/test
# Producto : flight_delay_model
# Lógica pura — no depende de Spark ni Databricks; testeable con pandas.
#
# REGLA: para series de tiempo NUNCA usar split aleatorio.
# Siempre usar split temporal para evitar data leakage.
# =============================================================================

from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitConfig:
    """
    Configuración del split temporal.

    TODO: ajustar los porcentajes según el volumen y horizonte del problema.
    Guía: train ~70%, val ~15%, test ~15%. Para series cortas: 60/20/20.
    """
    train_ratio: float = 0.70
    val_ratio:   float = 0.15
    # test_ratio se deriva: 1 - train_ratio - val_ratio

    def __post_init__(self):
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio debe ser < 1.0")


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    config: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en train/val/test respetando el orden temporal.

    Args:
        df:       DataFrame con columna de fecha.
        date_col: Nombre de la columna de fecha (debe ser datetime o string ISO).
        config:   Configuración de proporciones. Usa SplitConfig() por defecto.

    Returns:
        Tupla (df_train, df_val, df_test) ordenadas cronológicamente.

    Raises:
        ValueError: si date_col no existe o el DataFrame está vacío.
    """
    if config is None:
        config = SplitConfig()

    if date_col not in df.columns:
        raise ValueError(f"Columna de fecha '{date_col}' no encontrada en el DataFrame.")
    if df.empty:
        raise ValueError("El DataFrame está vacío.")

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * config.train_ratio)
    val_end   = int(n * (config.train_ratio + config.val_ratio))

    df_train = df_sorted.iloc[:train_end].copy()
    df_val   = df_sorted.iloc[train_end:val_end].copy()
    df_test  = df_sorted.iloc[val_end:].copy()

    return df_train, df_val, df_test


def split_summary(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    date_col: str,
) -> dict:
    """Devuelve un resumen del split para logging."""
    def date_range(df):
        if df.empty or date_col not in df.columns:
            return ("—", "—")
        return (str(df[date_col].min())[:10], str(df[date_col].max())[:10])

    return {
        "train": {"n": len(df_train), "range": date_range(df_train)},
        "val":   {"n": len(df_val),   "range": date_range(df_val)},
        "test":  {"n": len(df_test),  "range": date_range(df_test)},
    }
