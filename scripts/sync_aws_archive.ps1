param(
    [ValidateSet('Discover', 'Fetch', 'All')]
    [string]$Mode = 'All',
    [string]$ArchiveRoot = 'data/aws-official-archive',
    [string]$Policy = 'configs/corpus/aws-official-archive-v1.json',
    [ValidateRange(1, 10000)]
    [int]$BatchSize = 100,
    [ValidateRange(1, 86400)]
    [int]$SecondsPerBatch = 600,
    [ValidateRange(0, 1000000)]
    [int]$MaxBatches = 0
)

# Local downloads only. No AWS profile, model invocation, upload or deployment.
# Ctrl+C stops; rerunning resumes. No total document-count limit.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$archiveRepo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$archivePython = Join-Path $archiveRepo '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $archivePython -PathType Leaf)) {
    throw 'Run uv sync --frozen in the repository first.'
}

function Get-ArchiveCount($State, [string]$Kind, [string]$Name) {
    $archiveKind = $State.counts.PSObject.Properties[$Kind]
    if ($null -eq $archiveKind) { return 0 }
    $archiveCount = $archiveKind.Value.PSObject.Properties[$Name]
    if ($null -eq $archiveCount) { return 0 }
    return [int64]$archiveCount.Value
}

Push-Location -LiteralPath $archiveRepo
try {
    New-Item -ItemType Directory -Force -Path 'results/aws-archive' | Out-Null
    $archiveLatest = $null
    $archiveBatchesDone = 0
    $archiveJobs = @()
    if ($Mode -in @('Discover', 'All')) { $archiveJobs += 'SITEMAP' }
    if ($Mode -in @('Fetch', 'All')) { $archiveJobs += 'PAGE' }
    foreach ($archiveKindName in $archiveJobs) {
        $archiveCommand = if ($archiveKindName -eq 'SITEMAP') { 'archive-discover' } else { 'archive-fetch' }
        do {
            if ($MaxBatches -gt 0 -and $archiveBatchesDone -ge $MaxBatches) {
                Write-Host 'Per-run batch budget reached. Rerun to resume; corpus collection is not complete.'
                return
            }
            $archiveReport = 'results/aws-archive/latest-' + $archiveKindName.ToLowerInvariant() + '.json'
            & $archivePython -m app.cli $archiveCommand --root $ArchiveRoot --policy $Policy `
                --max-resources $BatchSize --max-seconds $SecondsPerBatch --report $archiveReport | Out-Host
            if ($LASTEXITCODE -ne 0) { throw ('Archive command failed: ' + $archiveCommand) }
            $archiveLatest = Get-Content -LiteralPath $archiveReport -Raw -Encoding UTF8 | ConvertFrom-Json
            $archiveBatchesDone += 1
            if ($archiveLatest.stop_reason -eq 'USER_STOP_REQUESTED') {
                Write-Host 'Archive paused by STOP file. Remove that file only when you want to resume.'
                return
            }
            if ($archiveLatest.stop_reason -notin @('QUEUE_DRAINED', 'RUN_BATCH_LIMIT', 'RUN_TIME_LIMIT')) {
                throw ('Archive stopped: ' + $archiveLatest.stop_reason + '. Read the report before resuming.')
            }
            $archiveQueued = (Get-ArchiveCount $archiveLatest $archiveKindName 'QUEUED') + `
                (Get-ArchiveCount $archiveLatest $archiveKindName 'FETCHING')
            if ($archiveQueued -gt 0 -and $archiveLatest.processed_this_run -eq 0) {
                throw 'No progress with pending resources. Inspect the latest report.'
            }
        } while ($archiveQueued -gt 0)
    }
    $archiveFailures = (Get-ArchiveCount $archiveLatest 'SITEMAP' 'FAILED') + `
        (Get-ArchiveCount $archiveLatest 'PAGE' 'FAILED')
    Write-Host ('Archive finished this run; failed resources: ' + $archiveFailures)
    Write-Host 'Read coverage flags and RAW_ONLY counts. This does not certify all AWS knowledge is present.'
    if ($archiveFailures -gt 0) { throw 'Some resources failed; use archive-status and explicit archive-retry after review.' }
}
finally {
    Pop-Location
}
