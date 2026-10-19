param(
    [string]$OutputPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'results/environment_preflight.json')
)

Set-StrictMode -Version Latest

function Invoke-ToolProbe {
    param([string]$Name, [string[]]$Arguments)

    $taskCommand = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $taskCommand) {
        return [pscustomobject]@{ tool = $Name; status = 'MISSING'; exit_code = $null; output = '' }
    }

    $taskPreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $taskRaw = & $taskCommand.Source @Arguments 2>&1
        $taskExit = $LASTEXITCODE
        $taskText = ($taskRaw | ForEach-Object { $_.ToString() }) -join "`n"
        return [pscustomobject]@{
            tool = $Name
            status = $(if ($taskExit -eq 0) { 'PASS' } else { 'FAIL' })
            exit_code = $taskExit
            output = $taskText.Trim()
        }
    }
    finally {
        $ErrorActionPreference = $taskPreviousPreference
    }
}

$taskProbes = @(
    (Invoke-ToolProbe -Name 'git' -Arguments @('--version'))
    (Invoke-ToolProbe -Name 'py' -Arguments @('-3.12', '--version'))
    (Invoke-ToolProbe -Name 'node' -Arguments @('--version'))
    (Invoke-ToolProbe -Name 'npm.cmd' -Arguments @('--version'))
    (Invoke-ToolProbe -Name 'uv' -Arguments @('--version'))
    (Invoke-ToolProbe -Name 'aws' -Arguments @('--version'))
    (Invoke-ToolProbe -Name 'docker' -Arguments @('--version'))
)
$taskDockerServer = Invoke-ToolProbe -Name 'docker' -Arguments @('info', '--format', '{{.ServerVersion}}')
$taskBlockers = @($taskProbes | Where-Object { $_.status -ne 'PASS' } | ForEach-Object { $_.tool })
if ($taskDockerServer.status -ne 'PASS') {
    $taskBlockers += 'docker_server'
}

$taskReport = [ordered]@{
    schema_version = 1
    checked_at = [DateTimeOffset]::UtcNow.ToOffset([TimeSpan]::FromHours(7)).ToString('o')
    scope = 'Local CLI availability only; no AWS APIs, credentials inspection, installation or inference.'
    status = $(if ($taskBlockers.Count -eq 0) { 'LOCAL_TOOLS_READY' } else { 'NOT_READY' })
    commands = $taskProbes
    docker_server = $taskDockerServer
    blockers = $taskBlockers
    package_compatibility = 'NOT_CHECKED'
    aws_identity = 'NOT_CHECKED'
    aws_service_and_model_access = 'NOT_CHECKED'
    application_tests = 'NOT_RUN'
}
$taskParent = Split-Path -Parent ([System.IO.Path]::GetFullPath($OutputPath))
New-Item -ItemType Directory -Path $taskParent -Force | Out-Null
$taskJson = $taskReport | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputPath), $taskJson + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
Write-Output $taskJson
if ($taskBlockers.Count -gt 0) { exit 1 }
exit 0
