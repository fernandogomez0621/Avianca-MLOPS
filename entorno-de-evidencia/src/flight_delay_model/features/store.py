# =============================================================================
# store.py — Wrapper sobre la feature table en Unity Catalog
# Producto : flight_delay_model
# =============================================================================

import os

DEFAULT_SCHEMA = os.environ.get("FEATURE_SCHEMA", "adv_dev_flight_delay")


def get_feature_table_name(catalog: str, schema: str | None = None) -> str:
    """Nombre completo de la feature table: {catalog}.{schema}.features"""
    return f"{catalog}.{schema or DEFAULT_SCHEMA}.features"


def publish_features(spark, catalog: str, features_df, schema: str | None = None) -> None:
    """
    Crea o actualiza la feature table en Unity Catalog.

    Usa FeatureEngineeringClient si esta disponible; si no (p.ej. serverless sin
    la libreria instalada), cae a una tabla Delta gobernada por UC con la misma
    estructura. El contrato hacia los consumidores no cambia.
    """
    table_name = get_feature_table_name(catalog, schema)
    primary_key = "client_id"

    try:
        from databricks.feature_engineering import FeatureEngineeringClient

        fe = FeatureEngineeringClient()
        fe.create_table(
            name=table_name,
            primary_keys=[primary_key],
            df=features_df,
            description=(
                "Feature table para flight_delay_model. "
                "Generada por features/notebooks/FeatureMaterialization.py."
            ),
        )
        print(f"[OK] Feature table (Feature Store) creada: {table_name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] FeatureEngineeringClient no disponible ({type(exc).__name__}): {exc}")
        print("[INFO] Publicando como tabla Delta gobernada por Unity Catalog.")
        (features_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(table_name))
        spark.sql(
            f"COMMENT ON TABLE {table_name} IS "
            "'Feature table de flight_delay_model (contrato definitions.py)'"
        )
        print(f"[OK] Feature table (Delta/UC) creada: {table_name}")
