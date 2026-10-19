param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('deepseek', 'voyage')]
    [string] $Provider,
    [switch] $Replace,
    [switch] $SelfTest
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$secretDirectory = Join-Path $projectRoot 'results\private\model-api-keys'
foreach ($relative in @('results', 'results\private', 'results\private\model-api-keys')) {
    $candidate = Join-Path $projectRoot $relative
    if (Test-Path -LiteralPath $candidate) {
        $item = Get-Item -LiteralPath $candidate -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw 'Secret directory must not contain a junction or symbolic link.'
        }
    }
}
[IO.Directory]::CreateDirectory($secretDirectory) | Out-Null
$currentSid = [Security.Principal.WindowsIdentity]::GetCurrent().User
$directoryAcl = New-Object Security.AccessControl.DirectorySecurity
$directoryAcl.SetOwner($currentSid)
$directoryAcl.SetAccessRuleProtection($true, $false)
$rule = New-Object Security.AccessControl.FileSystemAccessRule(
    $currentSid, [Security.AccessControl.FileSystemRights]::FullControl,
    ([Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [Security.AccessControl.InheritanceFlags]::ObjectInherit),
    [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow
)
$directoryAcl.AddAccessRule($rule)
# Persist only the modified owner/DACL; Set-Acl can request SACL privileges on rerun.
[IO.Directory]::SetAccessControl($secretDirectory, $directoryAcl)
$filename = if ($SelfTest) { 'self-test-' + [guid]::NewGuid().ToString('N') + '.json' } else { $Provider + '.json' }
$target = Join-Path $secretDirectory $filename
if (Test-Path -LiteralPath $target) {
    $existing = Get-Item -LiteralPath $target -Force
    if (($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw 'Secret file must not be a symbolic link.'
    }
    if (-not $Replace) { throw 'Key already exists. Use -Replace explicitly to rotate it.' }
}
$secure = $null
$pointer = [IntPtr]::Zero
$plain = $null
try {
    if ($SelfTest) {
        $plain = 'FIXTURE_ONLY_NOT_A_REAL_API_KEY_1234567890'
    } else {
        $secure = Read-Host "Paste $Provider API key (hidden input; press Enter)" -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    if ($plain.Length -lt 16 -or $plain.Length -gt 512 -or $plain -match '[^\x21-\x7e]') {
        throw 'Invalid key format. Expected 16..512 printable ASCII characters without spaces.'
    }
    $payload = @{ provider = $Provider; api_key = $plain } | ConvertTo-Json -Compress
    $bytes = (New-Object Text.UTF8Encoding($false)).GetBytes($payload)
    $mode = if ($Replace -and -not $SelfTest) { [IO.FileMode]::Create } else { [IO.FileMode]::CreateNew }
    $stream = [IO.File]::Open($target, $mode, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush() } finally { $stream.Dispose() }
    $acl = Get-Acl -LiteralPath $target
    $allows = @($acl.Access | Where-Object { $_.AccessControlType -eq 'Allow' })
    if ($allows.Count -lt 1) { throw 'Private ACL could not be verified.' }
    foreach ($access in $allows) {
        if ($access.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value -ne $currentSid.Value) {
            throw 'Unexpected principal can read the key. Storage verification failed.'
        }
    }
    if ($SelfTest) {
        Write-Output 'SELF_TEST_PASS: private ACL and exclusive write; no real key read or changed.'
    } else {
        Write-Output "STORED: $Provider key in project-private storage. No API request was made."
    }
} finally {
    if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    if ($null -ne $secure) { $secure.Dispose() }
    $plain = $null
    $payload = $null
    if ($SelfTest -and (Test-Path -LiteralPath $target)) { Remove-Item -LiteralPath $target -Force }
}
