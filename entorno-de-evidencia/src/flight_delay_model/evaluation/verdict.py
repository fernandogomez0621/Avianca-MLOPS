# =============================================================================
# verdict.py — Logica de veredicto PASS/FAIL (puro, sin Spark)
# Producto : flight_delay_model
# =============================================================================

from __future__ import annotations

from enum import Enum

from flight_delay_model.evaluation.thresholds import (
    CHAMPION_TOLERANCE,
    ITERATION_BUDGET,
    SLICE_THRESHOLDS,
    THRESHOLDS,
)


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class Hypothesis(str, Enum):
    """Indica a que skill volver cuando el veredicto es FAIL."""
    FEATURES         = "features"          # loop a S4
    HPO_ARCHITECTURE = "hpo_architecture"  # loop a S5
    DATA             = "data"              # loop a S2/S3


def check_absolute_thresholds(metrics: dict[str, float]) -> bool:
    """Verifica que todas las metricas cumplen los umbrales absolutos.

    Semantica maximize: la metrica debe ser >= al umbral definido.
    Aplica para clasificacion (auc, f1, recall, precision).
    Para metricas de error (rmse, mae) cambiar a: metrics.get(k, float("inf")) <= v
    """
    return all(metrics.get(k, 0.0) >= v for k, v in THRESHOLDS.items())


def check_champion_comparison(
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    primary_metric: str,
) -> bool:
    """Verifica no-inferioridad del candidato vs el champion actual.

    El candidato PASS si no es mas de CHAMPION_TOLERANCE peor que el champion.
    """
    if not champion_metrics:
        return True  # no hay champion — primer modelo siempre pasa
    champ_val = champion_metrics.get(primary_metric, 0.0)
    cand_val  = candidate_metrics.get(primary_metric, 0.0)
    # Semantica maximize: el candidato PASS si no cae mas de CHAMPION_TOLERANCE
    # por debajo del champion. Ejemplo: champ=0.80, tol=0.05 → cand >= 0.76
    return cand_val >= champ_val * (1 - CHAMPION_TOLERANCE)


def check_slice_thresholds(slice_metrics: dict[str, dict[str, float]]) -> bool:
    """Verifica que cada slice evaluado cumple su umbral propio.

    Solo se validan los slices presentes en slice_metrics.
    Si un slice de SLICE_THRESHOLDS no fue evaluado (sin datos), se omite.
    """
    for slice_name, thresholds in SLICE_THRESHOLDS.items():
        metrics = slice_metrics.get(slice_name)
        if metrics is None:
            continue   # slice sin datos suficientes — omitir
        for metric, threshold in thresholds.items():
            # Semantica maximize: FAIL si la metrica del slice < umbral
            if metrics.get(metric, 0.0) < threshold:
                return False
    return True


def classify_hypothesis(
    absolute_pass: bool,
    champion_pass: bool,
    slice_pass: bool,
    iteration_number: int,
) -> Hypothesis:
    """Clasifica la hipotesis de mejora para guiar el loop-back."""
    if not absolute_pass:
        if iteration_number <= 2:
            return Hypothesis.HPO_ARCHITECTURE
        return Hypothesis.FEATURES
    if not champion_pass:
        return Hypothesis.HPO_ARCHITECTURE
    if not slice_pass:
        return Hypothesis.FEATURES
    return Hypothesis.HPO_ARCHITECTURE


def generate_verdict(
    metrics: dict[str, float],
    champion_metrics: dict[str, float],
    slice_metrics: dict[str, dict[str, float]],
    primary_metric: str,
    iteration_number: int,
) -> dict:
    """Genera el veredicto completo PASS/FAIL con hipotesis e informe."""
    absolute_pass = check_absolute_thresholds(metrics)
    champion_pass = check_champion_comparison(metrics, champion_metrics, primary_metric)
    slice_pass    = check_slice_thresholds(slice_metrics)

    overall = Verdict.PASS if (absolute_pass and champion_pass and slice_pass) else Verdict.FAIL

    result = {
        "verdict": overall.value,
        "absolute_thresholds_pass": absolute_pass,
        "champion_comparison_pass": champion_pass,
        "slice_thresholds_pass": slice_pass,
        "iteration": iteration_number,
        "budget_remaining": ITERATION_BUDGET["max_iterations"] - iteration_number,
    }

    if overall == Verdict.FAIL:
        hypothesis = classify_hypothesis(
            absolute_pass, champion_pass, slice_pass, iteration_number
        )
        result["hypothesis"] = hypothesis.value

    return result


def decrement_budget(iteration_number: int) -> int:
    """Retorna el presupuesto restante y lanza error si se agoto."""
    remaining = ITERATION_BUDGET["max_iterations"] - iteration_number
    if remaining <= 0:
        raise RuntimeError(
            f"Presupuesto de iteracion agotado ({ITERATION_BUDGET['max_iterations']} iteraciones). "
            "Escalar a decision humana."
        )
    return remaining
