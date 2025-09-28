# PowerShell start script for Windows
# Usage: .\start.ps1 [options]
#   Options:
#     -Full        Start all services in Docker (default: infrastructure only)
#     -NoLogs      Don't show logs after starting

param(
    [switch]$Full,
    [switch]$NoLogs
)

Write-Host "🚀 Starting CocoIndex Development Environment..." -ForegroundColor Green

# Ensure logs directory exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

if ($Full) {
    Write-Host "📦 Starting all services in Docker containers..." -ForegroundColor Yellow
    docker-compose -f docker-compose.full.yml up -d

    Write-Host "⏳ Waiting for services to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10

    Write-Host "✅ All services started in Docker!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access points:" -ForegroundColor Cyan
    Write-Host "  - Backend API: http://localhost:8005"
    Write-Host "  - Frontend: http://localhost:3000"
    Write-Host "  - Neo4j Browser: http://localhost:7474"
    Write-Host "  - Qdrant Dashboard: http://localhost:6333/dashboard"
}
else {
    # Start infrastructure in Docker
    Write-Host "🗄️ Starting infrastructure services..." -ForegroundColor Yellow
    docker-compose -f docker-compose.dev.yml up -d

    Write-Host "⏳ Waiting for infrastructure to be healthy..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5

    # Find Python virtual environment
    $pythonCmd = $null
    $venvPaths = @(".venv312", "venv312", ".venv", "venv")
    foreach ($venv in $venvPaths) {
        $pythonPath = Join-Path $venv "Scripts\python.exe"
        if (Test-Path $pythonPath) {
            $pythonCmd = $pythonPath
            break
        }
    }

    if (-not $pythonCmd) {
        Write-Host "❌ No virtual environment found. Please create one first." -ForegroundColor Red
        Write-Host "   Run: python -m venv .venv312"
        exit 1
    }

    # Start backend
    Write-Host "🔧 Starting backend API..." -ForegroundColor Yellow
    $backendJob = Start-Process -FilePath $pythonCmd `
        -ArgumentList "-m", "uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8005" `
        -RedirectStandardOutput "logs\backend.log" `
        -RedirectStandardError "logs\backend.err" `
        -WindowStyle Hidden `
        -PassThru
    $backendJob.Id | Out-File "logs\backend.pid"

    # Start Celery worker
    Write-Host "👷 Starting Celery worker..." -ForegroundColor Yellow
    $celeryJob = Start-Process -FilePath $pythonCmd `
        -ArgumentList "-m", "celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--pool=solo" `
        -RedirectStandardOutput "logs\celery.log" `
        -RedirectStandardError "logs\celery.err" `
        -WindowStyle Hidden `
        -PassThru
    $celeryJob.Id | Out-File "logs\celery.pid"

    # Start frontend
    Write-Host "🎨 Starting frontend..." -ForegroundColor Yellow
    $frontendJob = Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "cd frontend && npm run dev" `
        -RedirectStandardOutput "logs\frontend.log" `
        -RedirectStandardError "logs\frontend.err" `
        -WindowStyle Hidden `
        -PassThru
    $frontendJob.Id | Out-File "logs\frontend.pid"

    Write-Host "✅ All services started!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access points:" -ForegroundColor Cyan
    Write-Host "  - Backend API: http://localhost:8005"
    Write-Host "  - Frontend: http://localhost:3000"
    Write-Host "  - Neo4j Browser: http://localhost:7474"
    Write-Host "  - Qdrant Dashboard: http://localhost:6333/dashboard"
    Write-Host ""
    Write-Host "Logs:" -ForegroundColor Cyan
    Write-Host "  - Backend: logs\backend.log"
    Write-Host "  - Celery: logs\celery.log"
    Write-Host "  - Frontend: logs\frontend.log"
}

if (-not $NoLogs) {
    Write-Host ""
    Write-Host "📋 Showing logs (Ctrl+C to stop viewing, services will continue running)..." -ForegroundColor Yellow
    if ($Full) {
        docker-compose -f docker-compose.full.yml logs -f
    }
    else {
        Get-Content logs\*.log -Wait
    }
}