param(
    [switch]$StopContainers,
    [int[]]$PortsToRelease = @(8005, 3000)
)

$ErrorActionPreference = 'SilentlyContinue'

$repoRoot = Split-Path -Parent $PSScriptRoot
$devLogs = Join-Path $repoRoot 'logs\dev'

function Write-Info($message) {
    Write-Host "[stop-dev] $message"
}

function Stop-ManagedProcess {
    param(
        [Parameter(Mandatory=$true)][string]$Name
    )

    $pidPath = Join-Path $devLogs "$Name.pid"
    if (-not (Test-Path $pidPath)) {
        Write-Info "$Name not tracked (no pid file)"
        return
    }

    $pid = Get-Content $pidPath | Select-Object -First 1
    if ($pid) {
        Write-Info "Stopping $Name (PID $pid)"
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
        } catch {
            Write-Info ("Unable to stop {0} by PID {1}: {2}" -f $Name, $pid, $_.Exception.Message)
        }
        Start-Sleep -Seconds 1
        if (Get-Process -Id $pid -ErrorAction SilentlyContinue) {
            Write-Info "$Name still running after initial stop, attempting force kill"
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }

    Remove-Item $pidPath -ErrorAction SilentlyContinue
}

if (Test-Path $devLogs) {
    Stop-ManagedProcess -Name 'frontend'
    Stop-ManagedProcess -Name 'celery-worker'
    Stop-ManagedProcess -Name 'backend'
} else {
    Write-Info 'No dev logs directory found; skipping managed process shutdown.'
}

if ($PortsToRelease) {
    foreach ($port in $PortsToRelease) {
        $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if (-not $listeners) { continue }
        $pids = $listeners | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique
        foreach ($pid in $pids) {
            if ($pid -and $pid -ne 0) {
                Write-Info "Stopping process $pid on port $port"
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

if ($StopContainers) {
    $containers = @(
        'cocoindex-redis',
        'cocoindex-qdrant',
        'cocoindex-neo4j',
        'cocoindex-postgres'
    )

    foreach ($name in $containers) {
        Write-Info "Stopping container $name"
        docker stop $name | Out-Null
    }
}

Write-Info 'Shutdown complete.'


