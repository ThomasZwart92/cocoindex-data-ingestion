param(
    [switch]$SkipContainers,
    [switch]$SkipBackend,
    [switch]$SkipWorker,
    [switch]$SkipFrontend,
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$logsRoot = Join-Path $repoRoot 'logs'
$devLogs = Join-Path $logsRoot 'dev'
if (-not (Test-Path $devLogs)) {
    New-Item -ItemType Directory -Path $devLogs -Force | Out-Null
}

function Write-Info($message) {
    Write-Host "[start-dev] $message"
}

$pythonPath = Join-Path $repoRoot '.venv312\Scripts\python.exe'
if (-not (Test-Path $pythonPath)) {
    Write-Info "Python interpreter not found at $pythonPath; falling back to system 'py'."
    $pythonPath = 'py'
}

function Test-PortsReady {
    param(
        [int[]]$Ports
    )

    foreach ($port in $Ports) {
        $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        if (-not $listener) {
            return $false
        }
    }
    return $true
}

function Start-ManagedProcess {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter()][string[]]$Arguments = @(),
        [Parameter(Mandatory=$true)][string]$WorkingDirectory,
        [Parameter()][int[]]$PortsToCheck,
        [int]$TimeoutSeconds = 30
    )

    $pidPath = Join-Path $devLogs "$Name.pid"
    $existingPid = $null
    if (Test-Path $pidPath) {
        $existingPid = (Get-Content $pidPath | Select-Object -First 1)
        if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
            Write-Info "$Name already running (PID $existingPid)"
            return
        } else {
            Remove-Item $pidPath -ErrorAction SilentlyContinue
            $existingPid = $null
        }
    }

    if ($PortsToCheck) {
        foreach ($port in $PortsToCheck) {
            $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
            if ($listener) {
                $owners = ($listener | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique)
                if ($owners -and $existingPid -and ($owners -contains [int]$existingPid)) {
                    Write-Info "$Name already listening on port $port (PID $existingPid)"
                    return
                }
                $ownerList = $owners -join ', '
                Write-Info "Port $port is already in use by PID(s) $ownerList. Skipping $Name start."
                return
            }
        }
    }

    $stdoutPath = Join-Path $devLogs "$Name.out"
    $stderrPath = Join-Path $devLogs "$Name.err"

    Write-Info "Starting $Name..."
    try {
        $process = Start-Process -FilePath $FilePath `
            -ArgumentList $Arguments `
            -WorkingDirectory $WorkingDirectory `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -WindowStyle Hidden `
            -PassThru
    } catch {
        Write-Info ("Failed to start {0}: {1}" -f $Name, $_.Exception.Message)
        return
    }

    $process.Id | Out-File -FilePath $pidPath -Encoding ascii -Force

    $deadline = (Get-Date).AddSeconds([Math]::Max($TimeoutSeconds, 5))
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        if ($process.HasExited) {
            Write-Info "$Name exited immediately with code $($process.ExitCode). Check $stderrPath"
            Remove-Item $pidPath -ErrorAction SilentlyContinue
            return
        }
        if (-not $PortsToCheck -or (Test-PortsReady -Ports $PortsToCheck)) {
            Write-Info "$Name started (PID $($process.Id))"
            return
        }
    }

    Write-Info "$Name did not report ready within $TimeoutSeconds seconds. Check $stderrPath for details."
}

if (-not $SkipContainers) {
    $containers = @(
        'cocoindex-redis',
        'cocoindex-qdrant',
        'cocoindex-neo4j',
        'cocoindex-postgres'
    )

    foreach ($name in $containers) {
        $status = docker ps --filter "name=$name" --format '{{.Status}}'
        if (-not $status) {
            Write-Info "Starting container $name..."
            docker start $name | Out-Null
        } else {
            Write-Info "Container $name already running ($status)"
        }
    }
}

if (-not $SkipBackend) {
    Start-ManagedProcess -Name 'backend' `
        -FilePath $pythonPath `
        -Arguments @('-m','uvicorn','app.main:app','--reload','--host','0.0.0.0','--port','8005') `
        -WorkingDirectory $repoRoot `
        -PortsToCheck @(8005) `
        -TimeoutSeconds $TimeoutSeconds
}

if (-not $SkipWorker) {
    Start-ManagedProcess -Name 'celery-worker' `
        -FilePath $pythonPath `
        -Arguments @('-m','celery','-A','app.tasks.document_tasks','worker','--loglevel=info') `
        -WorkingDirectory $repoRoot `
        -TimeoutSeconds $TimeoutSeconds
}

if (-not $SkipFrontend) {
    Start-ManagedProcess -Name 'frontend' `
        -FilePath 'npm.cmd' `
        -Arguments @('run','dev') `
        -WorkingDirectory (Join-Path $repoRoot 'frontend') `
        -PortsToCheck @(3000) `
        -TimeoutSeconds $TimeoutSeconds
}

Write-Info 'All requested services processed.'




