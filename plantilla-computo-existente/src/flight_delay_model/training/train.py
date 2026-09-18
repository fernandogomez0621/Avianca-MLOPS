# =============================================================================
# train.py — Logica pura de entrenamiento (sin Spark, testeable con pytest)
# Producto : flight_delay_model | Dominio: ops | Area: operaciones
# =============================================================================

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Columnas — completar con las features reales del modelo
# -----------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "dep_hour",
    "day_of_week",
    "month",
    "carrier",
    "distance_km",
    "prev_delay",
    "congestion",
]

TARGET_COL: str = "target"   # 1 = retraso de llegada >= 15 min

# -----------------------------------------------------------------------------
# Hiperparametros por defecto
# -----------------------------------------------------------------------------
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": 42,
}


def create_pipeline(params: dict[str, Any]) -> Pipeline:
    """Crea el pipeline de preprocesamiento + modelo.

    Args:
        params: Hiperparametros del modelo.

    Returns:
        Pipeline de scikit-learn listo para fit.
    """
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(**params),
    )


def train_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Entrena el pipeline sobre los datos de training.

    Args:
        pipeline: Pipeline creado por create_pipeline.
        X: Features de entrenamiento.
        y: Target de entrenamiento.

    Returns:
        Pipeline entrenado (fitted).
    """
    return pipeline.fit(X, y)


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calcula las metricas principales del modelo sobre un conjunto.

    Args:
        model: Pipeline entrenado.
        X: Features del conjunto a evaluar.
        y: Target real.

    Returns:
        Diccionario con metricas (al menos las declaradas en thresholds.py).
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return {
        "auc":       float(roc_auc_score(y, proba)),
        "f1":        float(f1_score(y, preds, zero_division=0)),
        "recall":    float(recall_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
    }


def evaluate_slice(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    slice_col: str,
    slice_val: Any,
) -> dict[str, float]:
    """Calcula metricas para un slice especifico de negocio.

    Args:
        model: Pipeline entrenado.
        X: Features completas (con la columna de slice).
        y: Target completo.
        slice_col: Nombre de la columna por la que se filtra.
        slice_val: Valor del slice.

    Returns:
        Metricas del modelo sobre ese slice.
    """
    mask = X[slice_col] == slice_val
    if mask.sum() == 0:
        return {}
    return evaluate_model(model, X[mask], y[mask])


def check_thresholds(
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> bool:
    """Verifica si todas las metricas pasan sus umbrales.

    Args:
        metrics: Metricas calculadas por evaluate_model.
        thresholds: Umbrales maximos (o minimos) de aceptacion.

    Returns:
        True si todos los umbrales se cumplen.
    """
    # Semantica maximize (clasificacion): metrica debe ser >= umbral
    # Para metricas de error (rmse, mae) cambiar a: metrics.get(k, float("inf")) <= v
    return all(metrics.get(k, 0.0) >= v for k, v in thresholds.items())


def get_feature_importance(model: Pipeline) -> dict[str, float]:
    """Extrae la importancia de features del modelo entrenado.

    Returns:
        Diccionario {feature_name: importance_score}.
    """
    # TODO: implementar segun el tipo de estimador
    # estimator = model.named_steps.get("lgbmregressor") or model[-1]
    # return dict(zip(FEATURE_COLS, estimator.feature_importances_))
    return {}
