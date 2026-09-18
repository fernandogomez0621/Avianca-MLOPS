# =============================================================================
# thresholds.py — Configuracion de umbrales de evaluacion (puro, sin Spark)
# Producto : flight_delay_model
# =============================================================================

from __future__ import annotations

# -----------------------------------------------------------------------------
# Umbrales absolutos — el modelo debe cumplir TODOS para PASS
# TODO: ajustar con los criterios de aceptacion del negocio
# -----------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    "auc":       0.70,
    "f1":        0.55,
    "recall":    0.55,
    "precision": 0.50,
}

# Tolerancia para comparacion no-inferioridad vs champion (porcentaje)
# El candidato puede ser hasta CHAMPION_TOLERANCE peor que el champion y aun PASS
CHAMPION_TOLERANCE: float = 0.02   # 2% — margen de no-inferioridad vs champion

# -----------------------------------------------------------------------------
# Umbrales por slice de negocio
# TODO: definir los segmentos criticos del negocio y sus umbrales
# -----------------------------------------------------------------------------
SLICE_THRESHOLDS: dict[str, dict[str, float]] = {
    "hora_pico":     {"auc": 0.65, "recall": 0.50},
    "hora_valle":    {"auc": 0.65, "recall": 0.50},
    "fin_de_semana": {"auc": 0.65, "recall": 0.50},
}

# Mapeo slice -> (columna, valor) para calcular metricas por segmento.
SLICE_DEFINITIONS: dict[str, tuple[str, int]] = {
    "hora_pico":     ("dep_hour", 18),
    "hora_valle":    ("dep_hour", 6),
    "fin_de_semana": ("day_of_week", 6),
}

# -----------------------------------------------------------------------------
# Presupuesto de iteracion — cuantas veces puede fallar antes de escalar
# -----------------------------------------------------------------------------
ITERATION_BUDGET: dict[str, int] = {
    "max_iterations": 5,
}
