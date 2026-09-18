# =============================================================================
# init.ps1 — Inicializador de plantilla ML BPT
# Uso: .\init.ps1 -ProductName "mi_modelo" -Domain "ops" -AreaNegocio "scm"
#                 -Proyecto "demand_forecasting" -ServicePrincipal "sp-<dominio>-prd@<organizacion>"
#                 -WorkspaceHost "https://adb-xxxxx.azuredatabricks.net"
#                 -ClusterIdDev "xxxx-xxxxxx-xxxxxxxx" -ClusterIdQa "xxxx-xxxxxx-xxxxxxxx"
#                 -ClusterIdPrd "xxxx-xxxxxx-xxxxxxxx"
#                 -ClusterIdMaterialization "xxxx-xxxxxx-xxxxxxxx"
#
# Nota: ClusterIdMaterialization es un cluster NO-ML dedicado exclusivamente al job
#       de feature materialization (los clusters ML de training no pueden ejecutarlo).
#       Un solo valor se usa en todos los ambientes (dev/test/qa/prod).
# =============================================================================

param(
    [Parameter(Mandatory)][string]$ProductName,       # snake_case: demand_forecast_model
    [Parameter(Mandatory)][string]$Domain,            # ops | retail | fin | hr
    [Parameter(Mandatory)][string]$AreaNegocio,       # scm | pricing | rrhh
    [Parameter(Mandatory)][string]$Proyecto,          # demand_forecasting
    [Parameter(Mandatory)][string]$ServicePrincipal,  # sp-<dominio>-prd@<organizacion> (o GUID del SP de prod)
    [Parameter(Mandatory)][string]$ServiceConnection, # nombre del ARM Service Connection en Azure DevOps
    [string]$ModelName     = "",                      # md_scm_demand (se deriva si no se pasa)
    [string]$WorkspaceHost = "",                      # https://adb-xxxxx.azuredatabricks.net
    [string]$ClusterIdDev  = "",                      # ID cluster existente en DEV
    [string]$ClusterIdQa   = "",                      # ID cluster existente en QA
    [string]$ClusterIdPrd  = "",                      # ID cluster existente en PRD
    [string]$ClusterIdMaterialization = ""             # ID cluster NO-ML para feature materialization (todos los ambientes)
)

$ErrorActionPreference = "Stop"

# Derivar variables
$ProductNameDash = $ProductName -replace "_", "-"    # demand-forecast-model
if (-not $ModelName) {
    $ModelName = "md_$($AreaNegocio)_$(($Proyecto -split '_')[0])"  # md_scm_demand
}
if (-not $WorkspaceHost) { $WorkspaceHost = "https://your-workspace.azuredatabricks.net" }
if (-not $ClusterIdDev)  { $ClusterIdDev  = "TODO_CLUSTER_ID_DEV" }
if (-not $ClusterIdQa)   { $ClusterIdQa   = "TODO_CLUSTER_ID_QA"  }
if (-not $ClusterIdPrd)  { $ClusterIdPrd  = "TODO_CLUSTER_ID_PRD" }
if (-not $ClusterIdMaterialization) { $ClusterIdMaterialization = "TODO_CLUSTER_ID_MATERIALIZATION" }

Write-Host ""
Write-Host "=== Inicializando plantilla ML BPT ===" -ForegroundColor Cyan
Write-Host "  ProductName      : $ProductName"
Write-Host "  ProductNameDash  : $ProductNameDash"
Write-Host "  Domain           : $Domain"
Write-Host "  AreaNegocio      : $AreaNegocio"
Write-Host "  Proyecto         : $Proyecto"
Write-Host "  ServicePrincipal : $ServicePrincipal"
Write-Host "  ServiceConnection: $ServiceConnection"
Write-Host "  ModelName        : $ModelName"
Write-Host "  WorkspaceHost    : $WorkspaceHost"
Write-Host "  ClusterIdDev     : $ClusterIdDev"
Write-Host "  ClusterIdQa      : $ClusterIdQa"
Write-Host "  ClusterIdPrd     : $ClusterIdPrd"
Write-Host "  ClusterIdMaterialization : $ClusterIdMaterialization  (NO-ML, todos los ambientes)"
Write-Host ""

# 1 — Renombrar carpeta src\PRODUCT_NAME
$srcOld = Join-Path $PSScriptRoot "src\PRODUCT_NAME"
$srcNew = Join-Path $PSScriptRoot "src\$ProductName"

if (Test-Path $srcOld) {
    Rename-Item -Path $srcOld -NewName $ProductName
    Write-Host "[OK] Renombrado src\PRODUCT_NAME -> src\$ProductName" -ForegroundColor Green
} elseif (Test-Path $srcNew) {
    Write-Host "[SKIP] src\$ProductName ya existe" -ForegroundColor Yellow
} else {
    Write-Host "[ERROR] No se encontro src\PRODUCT_NAME ni src\$ProductName" -ForegroundColor Red
    exit 1
}

# 2 — Reemplazar placeholders en todos los archivos de texto
# IMPORTANTE: el orden es critico — los mas especificos primero para evitar reemplazos parciales.
# NOTA: se usa -creplace (case-sensitive) para evitar que SERVICE_PRINCIPAL
#       reemplace tambien "service_principal" en claves YAML, y DOMAIN reemplace "domain".
$extensions = @("*.py", "*.yml", "*.yaml", "*.toml", "*.txt", "*.md", "*.json")
# NOTA: NO usar "Select-Object -Unique" aqui — FileInfo.ToString() devuelve solo el
# Name (no el FullName), asi que -Unique deduplicaria por nombre de archivo y
# descartaria silenciosamente archivos con el mismo nombre en carpetas distintas
# (p.ej. __init__.py o validation.py, que existen en varias subcarpetas), dejando
# placeholders sin reemplazar en esos duplicados. Cada patron de extension ya
# devuelve rutas distintas, por lo que no se requiere deduplicacion.
$files = $extensions | ForEach-Object { Get-ChildItem -Path $PSScriptRoot -Filter $_ -Recurse }

$replacements = [ordered]@{
    "PRODUCT_NAME_DASH"      = $ProductNameDash     # primero — mas especifico
    "MODEL_NAME_PLACEHOLDER" = $ModelName
    "CLUSTER_ID_DEV"         = $ClusterIdDev
    "CLUSTER_ID_QA"          = $ClusterIdQa
    "CLUSTER_ID_PRD"         = $ClusterIdPrd
    "CLUSTER_ID_MATERIALIZATION" = $ClusterIdMaterialization
    "WORKSPACE_HOST"         = $WorkspaceHost
    "PRODUCT_NAME"           = $ProductName
    "SERVICE_PRINCIPAL"      = $ServicePrincipal
    "SERVICE_CONNECTION"     = $ServiceConnection
    "AREA_NEGOCIO"           = $AreaNegocio
    "PROYECTO"               = $Proyecto
    "DOMAIN"                 = $Domain
}

$enc = [System.Text.UTF8Encoding]::new($false)   # UTF8 sin BOM
$changedCount = 0
foreach ($file in $files) {
    # ReadAllText es mas robusto que Get-Content en PS 5.1 con archivos UTF8
    $content = [System.IO.File]::ReadAllText($file.FullName, $enc)
    $newContent = $content
    foreach ($key in $replacements.Keys) {
        $newContent = $newContent -creplace $key, $replacements[$key]   # -creplace = case-sensitive
    }
    if ($newContent -ne $content) {
        [System.IO.File]::WriteAllText($file.FullName, $newContent, $enc)
        $relPath = $file.FullName.Replace($PSScriptRoot, "").TrimStart("\")
        Write-Host "[OK] $relPath" -ForegroundColor Green
        $changedCount++
    }
}

Write-Host ""
Write-Host "=== Completado: $changedCount archivos actualizados ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Siguientes pasos:" -ForegroundColor Yellow
Write-Host "  1. Completar los TODOs en src\$ProductName\training\train.py"
Write-Host "  2. Completar los umbrales en src\$ProductName\evaluation\thresholds.py"
Write-Host "  3. Definir features en src\$ProductName\features\definitions.py"
Write-Host "  4. Los variable groups son genericos: mlops-vg-ci / mlops-vg-staging / mlops-vg-prd"
Write-Host "  5. databricks bundle validate --target dev"
