# PowerShell stop script for Windows
# Usage: .\stop.ps1 [options]
#   Options:
#     -Full        Stop Docker containers too (default: only stops host services)
#     -Clean       Remove volumes and clean data

param(
    [switch]$Full,
    [switch]$Clean
)

Write-Host "🛑 Stopping CocoIndex services..." -ForegroundColor Yellow

# Stop host-based services using PID files
if (Test-Path "logs") {
    Get-ChildItem "logs\*.pid" -ErrorAction SilentlyContinue | ForEach-Object {
        $serviceName = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
        $pid = Get-Content $_.FullName -First 1

        try {
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process) {
                Write-Host "  Stopping $serviceName (PID: $pid)..." -ForegroundColor Gray
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 1
            }
        }
        catch {
            # Process already stopped
        }

        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
    }
}

# Stop any orphaned processes on known ports
$ports = @(8005, 8001, 3000)
foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($connections) {
        $pids = $connections | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($pid in $pids) {
            if ($pid -gt 0) {
                Write-Host "  Stopping process on port $port (PID: $pid)..." -ForegroundColor Gray
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

# Kill any remaining node processes (frontend)
Get-Process node -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*cocoindex*"
} | ForEach-Object {
    Write-Host "  Stopping Node.js process (PID: $($_.Id))..." -ForegroundColor Gray
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

# Kill any remaining Python processes (backend/celery)
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -like "*cocoindex*"
} | ForEach-Object {
    Write-Host "  Stopping Python process (PID: $($_.Id))..." -ForegroundColor Gray
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

if ($Full -or $Clean) {
    Write-Host "🐳 Stopping Docker containers..." -ForegroundColor Yellow

    # Try all compose files to ensure we stop everything
    docker-compose -f docker-compose.dev.yml down 2>$null
    docker-compose -f docker-compose.full.yml down 2>$null
    docker-compose down 2>$null

    if ($Clean) {
        Write-Host "🗑️ Cleaning up volumes and data..." -ForegroundColor Yellow
        docker-compose -f docker-compose.dev.yml down -v 2>$null
        docker-compose -f docker-compose.full.yml down -v 2>$null

        # Clean log files
        Remove-Item logs\*.log -Force -ErrorAction SilentlyContinue
        Remove-Item logs\*.err -Force -ErrorAction SilentlyContinue
        Remove-Item logs\*.pid -Force -ErrorAction SilentlyContinue

        Write-Host "  All data cleaned!" -ForegroundColor Green
    }
}

Write-Host "✅ All services stopped!" -ForegroundColor Green