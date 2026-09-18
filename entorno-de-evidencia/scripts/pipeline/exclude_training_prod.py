#!/usr/bin/env python3
"""
exclude_training_prod.py — Excluye el job de training del deploy a prod.

Por que: el include de model-training-workflow.yml es global (aplica a todos los targets:
dev/test/qa/prod). Sin este script, "databricks bundle deploy --target prod" tambien crea
el job de training ahi — sin schedule propio, pero disponible para correr manualmente, lo
que permitiria registrar un modelo directo en el catalogo de PRD (via notebook_promote.py,
catalog=${var.catalog_name} = gold_adv_prd) saltandose la promocion via copy_model_version
y el gate de aprobacion manual que protege el paso a produccion.

Comenta la linea de include marcada con "# [CD-TRAINING-EXCLUDE-PRD]" en databricks.yml,
quitandola del bundle SOLO para el deploy que sigue. Ejecutado por cd-scoring-pipeline
INMEDIATAMENTE ANTES de cada "databricks bundle deploy --target prod" (stages DeployPrd y
RunScoringPrd — este ultimo redespliega para activar el monitor). Cada stage de Azure DevOps
corre en un agente nuevo con checkout limpio del repo, por lo que el script debe correr antes
de CADA deploy a prod, no solo el primero.

Uso local (fuera del pipeline):
    python3 scripts/pipeline/exclude_training_prod.py databricks.yml
    databricks bundle deploy --target prod
    git checkout -- databricks.yml   # restaura el include para dev/test/qa

El archivo se modifica en sitio. Idempotente: seguro de ejecutar sobre un YAML ya excluido.
"""
import re
import sys
from pathlib import Path

filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("databricks.yml")

if not filepath.exists():
    print(f"[ERROR] No se encontro {filepath}", file=sys.stderr)
    sys.exit(1)

MARKER = "[CD-TRAINING-EXCLUDE-PRD]"
# Linea de include ACTIVA (sin # al inicio) que lleva el marker al final.
active_pattern = re.compile(
    r"^(\s*)(- \./resources/model-training-workflow\.yml.*" + re.escape(MARKER) + r".*)$",
    re.MULTILINE,
)

text = filepath.read_text(encoding="utf-8")
matches = active_pattern.findall(text)

if not matches:
    if MARKER in text:
        print(f"[OK] exclude_training_prod: ya estaba excluido en {filepath} (idempotente)")
        sys.exit(0)
    print(f"[ERROR] No se encontro la linea de include con marker {MARKER} en {filepath}", file=sys.stderr)
    sys.exit(1)

text_out = active_pattern.sub(r"\1# \2", text)
filepath.write_text(text_out, encoding="utf-8")
print(f"[OK] exclude_training_prod: {len(matches)} linea(s) excluida(s) en {filepath}")
