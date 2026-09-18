# =============================================================================
# hpo.py — Hyperparameter optimization gobernado con presupuesto (Optuna)
# Producto : flight_delay_model
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from flight_delay_model.training.train import create_pipeline, evaluate_model, train_model

if TYPE_CHECKING:
    # optuna se importa de forma diferida en run_hpo() (no disponible fuera de
    # Databricks) — este import solo existe para el type hint de _suggest_params.
    import optuna

# Presupuesto declarado — nunca HPO sin presupuesto
BUDGET: dict[str, Any] = {
    "max_trials": 30,       # TODO: ajustar segun tiempo disponible
    "max_minutes": 60,
    "max_dbu": 4.0,
}

# TODO: definir el espacio de busqueda segun el algoritmo elegido
SEARCH_SPACE: dict[str, Any] = {
    # Ejemplo para LightGBM:
    # "n_estimators":  {"type": "int",   "low": 100,  "high": 500},
    # "learning_rate": {"type": "float", "low": 0.01, "high": 0.3,  "log": True},
    # "max_depth":     {"type": "int",   "low": 3,    "high": 9},
    # "num_leaves":    {"type": "int",   "low": 20,   "high": 100},
}


def validate_budget(budget: dict[str, Any]) -> None:
    """Verifica que el presupuesto tiene todos los campos requeridos."""
    required = {"max_trials", "max_minutes", "max_dbu"}
    missing = required - set(budget.keys())
    if missing:
        raise ValueError(f"BUDGET incompleto — faltan campos: {missing}")


def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sugiere hiperparametros desde el espacio de busqueda declarado."""
    params: dict[str, Any] = {}
    for name, cfg in SEARCH_SPACE.items():
        if cfg["type"] == "int":
            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
        elif cfg["type"] == "float":
            params[name] = trial.suggest_float(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
            )
        elif cfg["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, cfg["choices"])
    return params


def run_hpo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, Any], float]:
    """Ejecuta HPO con Optuna dentro del presupuesto declarado.

    Cada trial es un run hijo en MLflow (trazabilidad completa).

    Args:
        X_train, y_train: Datos de entrenamiento.
        X_val, y_val: Datos de validacion para la funcion objetivo.

    Returns:
        (best_params, best_metric_value)
    """
    validate_budget(BUDGET)

    import mlflow  # lazy import — no disponible fuera de Databricks
    import optuna

    # TODO: definir la metrica objetivo y la direccion (minimize / maximize)
    PRIMARY_METRIC = "rmse"  # debe coincidir con evaluate_model()
    DIRECTION = "minimize"

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        with mlflow.start_run(run_name=f"hpo_trial_{trial.number}", nested=True):
            mlflow.log_params(params)
            model = train_model(create_pipeline(params), X_train, y_train)
            metrics = evaluate_model(model, X_val, y_val)
            mlflow.log_metrics(metrics)
        return metrics[PRIMARY_METRIC]

    study = optuna.create_study(direction=DIRECTION)
    study.optimize(
        objective,
        n_trials=BUDGET["max_trials"],
        timeout=BUDGET["max_minutes"] * 60,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_value = study.best_value
    return best_params, best_value
