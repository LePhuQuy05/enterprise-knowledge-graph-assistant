param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$kgRepo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$kgResults = Join-Path $kgRepo 'results/lambda_runtime_smoke'
New-Item -ItemType Directory -Path $kgResults -Force | Out-Null
$kgRun = Join-Path $kgResults ('run-' + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $kgRun | Out-Null
$kgImages = Get-Content (Join-Path $kgRepo 'configs/runtime_images.json') -Raw | ConvertFrom-Json
$kgRequirements = Join-Path $kgRun 'requirements.txt'
Push-Location $kgRepo
try {
    uv export --quiet --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt --output-file $kgRequirements
    if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency export failed' }
    # Mount only source code and this isolated output. No home/.aws or environment files.
    docker run --rm --platform linux/amd64 --network bridge `
        --mount "type=bind,source=$kgRepo\app,target=/workspace/app,readonly" `
        --mount "type=bind,source=$PSScriptRoot\check_lambda_runtime.py,target=/workspace/smoke.py,readonly" `
        --mount "type=bind,source=$kgRun,target=/asset" `
        --entrypoint /bin/sh $kgImages.lambda_python `
        -c 'python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: --no-compile --target /tmp/kg-runtime -r /asset/requirements.txt && KG_RUNTIME_DEPS_DIR=/tmp/kg-runtime python /workspace/smoke.py'
    if ($LASTEXITCODE -ne 0) { throw 'Linux dependency/import smoke failed; inspect the isolated output' }
    $kgReportPath = Join-Path $kgRun 'report.json'
    $kgReport = Get-Content $kgReportPath -Raw | ConvertFrom-Json
    $kgReport | Add-Member -NotePropertyName image -NotePropertyValue $kgImages.lambda_python
    $kgReport | Add-Member -NotePropertyName lock_sha256 -NotePropertyValue (Get-FileHash (Join-Path $kgRepo 'uv.lock') -Algorithm SHA256).Hash.ToLowerInvariant()
    $kgReport | Add-Member -NotePropertyName checked_at -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString('o'))
    $kgReportJson = $kgReport | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($kgReportPath, $kgReportJson + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Write-Output "Linux smoke report: $kgReportPath"
}
finally {
    Pop-Location
}
