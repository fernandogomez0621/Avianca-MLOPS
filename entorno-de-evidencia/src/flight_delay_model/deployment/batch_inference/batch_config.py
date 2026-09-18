# =============================================================================
# batch_config.py — Configuración y metadata del scoring batch
# Producto : flight_delay_model
# =============================================================================

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass
class BatchConfig:
    """Configuración de una ejecución de scoring batch."""
    catalog:     str
    model_name:  str
    start_date:  str = ""
    end_date:    str = ""

    def __post_init__(self):
        if not self.start_date:
            self.start_date = str(date.today() - timedelta(days=1))
        if not self.end_date:
            self.end_date = str(date.today())

    @property
    def env(self) -> str:
        """Deriva el entorno desde el catálogo: gold_retail_dev → dev."""
        return self.catalog.split("_")[-1]

    @property
    def model_fqn(self) -> str:
        """Nombre completo del modelo en UC: catalog.schema.model_name."""
        return f"{self.catalog}.operaciones_flight_delay.{self.model_name}"

    @property
    def model_uri(self) -> str:
        """URI del alias @champion."""
        return f"models:/{self.model_fqn}@champion"


def get_inference_table_name(config: BatchConfig) -> str:
    """Devuelve el nombre completo de la inference table."""
    return f"{config.catalog}.operaciones_flight_delay.{config.model_name}_inference"


def build_inference_metadata(config: BatchConfig, n_records: int) -> dict:
    """Construye el diccionario de metadata para loguear en MLflow o Delta."""
    return {
        "scored_at":   str(date.today()),
        "model_uri":   config.model_uri,
        "start_date":  config.start_date,
        "end_date":    config.end_date,
        "n_records":   n_records,
        "env":         config.env,
    }
