param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$kgRepo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$kgBaseline = Get-Content -LiteralPath (Join-Path $kgRepo 'results/bge_baseline.json') -Raw | ConvertFrom-Json
$kgImage = (Get-Content -LiteralPath (Join-Path $kgRepo 'configs/runtime_images.json') -Raw | ConvertFrom-Json).lambda_python
$kgAsset = Join-Path $kgRepo 'results/bge-linux'
New-Item -ItemType Directory -Path $kgAsset -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $kgRepo 'results/bge-linux-requirements.txt') -Destination (Join-Path $kgAsset 'requirements.txt')
# Source/model/snapshot are read-only; no .aws, home, credentials or broad repo/results mount.
docker run --rm --platform linux/amd64 --network bridge `
    --mount "type=bind,source=$kgRepo\app,target=/workspace/app,readonly" `
    --mount "type=bind,source=$kgRepo\scripts\check_bge_linux.py,target=/workspace/smoke.py,readonly" `
    --mount "type=bind,source=$kgRepo\configs\models\bge-small-en-v1.5-onnx.json,target=/workspace/bge.json,readonly" `
    --mount "type=bind,source=$kgRepo\data\models\bge-small-en-v1.5,target=/model,readonly" `
    --mount "type=bind,source=$($kgBaseline.snapshot_directory),target=/snapshot,readonly" `
    --mount "type=bind,source=$kgAsset,target=/asset" `
    --entrypoint /bin/sh $kgImage `
    -c 'python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: --no-compile --target /tmp/kg-bge -r /asset/requirements.txt && PYTHONPATH=/tmp/kg-bge:/workspace python /workspace/smoke.py'
if ($LASTEXITCODE -ne 0) { throw 'BGE Linux smoke failed; inspect output before deployment' }
Write-Output 'BGE Linux report: results/bge-linux/report.json'
