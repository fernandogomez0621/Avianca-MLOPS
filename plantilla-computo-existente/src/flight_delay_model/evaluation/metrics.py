# =============================================================================
# metrics.py — Métricas custom para mlflow.evaluate
# Producto : flight_delay_model
# Lógica pura — no depende de Spark; testeable con pandas/numpy.
# =============================================================================

import numpy as np

# TODO: seleccionar las métricas adecuadas según el tipo de problema:
#   Regresión:      MAE, RMSE, MAPE, R²
#   Clasificación:  F1, Precision, Recall, ROC-AUC, PR-AUC
#   Series de tiempo: MASE, sMAPE, cobertura de intervalos

# ---------------------------------------------------------------------------
# Métricas de regresión (comentar si el modelo es clasificación)
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE. Excluye ceros en y_true para evitar división por cero."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


# ---------------------------------------------------------------------------
# Métricas de clasificación (comentar si el modelo es regresión)
# ---------------------------------------------------------------------------

def f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """F1 para clasificación binaria. y_pred puede ser probabilidad o clase."""
    from sklearn.metrics import f1_score
    y_bin = (y_pred >= threshold).astype(int) if y_pred.dtype == float else y_pred
    return float(f1_score(y_true, y_bin, zero_division=0))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


# ---------------------------------------------------------------------------
# Registro para mlflow.evaluate (seleccionar las apropiadas)
# ---------------------------------------------------------------------------

# TODO: descomentar las métricas que aplican al modelo y registrarlas en Train.py
# con mlflow.evaluate(..., extra_metrics=[...])
# Ejemplo:
# from mlflow.models import make_metric
# CUSTOM_METRICS = [
#     make_metric(eval_fn=mean_absolute_percentage_error, name="mape", greater_is_better=False),
#     make_metric(eval_fn=f1_score_binary, name="f1_binary", greater_is_better=True),
# ]
CUSTOM_METRICS = []  # TODO: completar con las métricas del modelo
