#!/usr/bin/env python3
"""
enable_monitor.py — Activa las secciones de quality_monitors en databricks.yml.
Ejecutado por cd-scoring-pipeline tras la primera ejecucion del batch scoring job.

Busca lineas marcadas con "# [CD-MONITOR-{TARGET}] " y les quita el prefijo de comentario,
transformando el YAML comentado en YAML activo. Cada target (qa, prod) tiene sus propios
markers para que la activacion sea independiente — activar QA no activa PRD ni viceversa.

NOTA: ejecutar siempre DESPUES de que el batch scoring job haya corrido al menos una vez,
ya que el monitor requiere que la tabla _inference exista previamente.

Uso:
    python3 scripts/pipeline/enable_monitor.py [path/to/databricks.yml] [--target qa|prod]

    --target qa    activa markers # [CD-MONITOR-QA]  (default)
    --target prod  activa markers # [CD-MONITOR-PRD]

El archivo se modifica en sitio. Idempotente: seguro de ejecutar sobre un YAML ya activo.
"""
import re
import sys
from pathlib import Path


_TARGET_MARKER_ALIASES = {
    "PROD": "PRD",  # el target se llama "prod" pero el marker en databricks.yml es "PRD"
}


def get_marker(target: str) -> str:
    """Retorna el marker de comentario para el target indicado."""
    normalized = _TARGET_MARKER_ALIASES.get(target.upper(), target.upper())
    return f"# [CD-MONITOR-{normalized}] "


# Parsear argumentos
filepath = Path("databricks.yml")
target = "qa"

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--target" and i + 1 < len(args):
        target = args[i + 1]
        i += 2
    elif not args[i].startswith("--"):
        filepath = Path(args[i])
        i += 1
    else:
        i += 1

if not filepath.exists():
    print(f"[ERROR] No se encontro {filepath}", file=sys.stderr)
    sys.exit(1)

marker = get_marker(target)
pattern = re.compile(r"^(\s*)" + re.escape(marker), re.MULTILINE)

text = filepath.read_text(encoding="utf-8")
count = len(pattern.findall(text))
text_out = pattern.sub(r"\1", text)

filepath.write_text(text_out, encoding="utf-8")
print(f"[OK] enable_monitor --target {target}: {count} lineas activadas en {filepath}")
