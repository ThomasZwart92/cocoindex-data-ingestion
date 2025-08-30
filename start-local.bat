@echo off
REM CocoIndex Local Deployment Script for Windows
REM This script starts all services locally using Docker Compose

echo Starting CocoIndex Local Deployment...

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found!
    echo Please copy .env.example to .env and configure your API keys
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running!
    echo Please start Docker Desktop and try again
    exit /b 1
)

REM Stop any existing containers
echo Stopping existing containers...
docker-compose -f docker-compose.yml down 2>nul

REM Start infrastructure services
echo Starting infrastructure services...
docker-compose -f docker-compose.yml up -d

REM Wait for services to be healthy
echo Waiting for services to be healthy...
timeout /t 10 /nobreak >nul

REM Check service health
echo Checking service health...
docker-compose -f docker-compose.yml ps

REM Initialize CocoIndex
echo Initializing CocoIndex...
python -c "import cocoindex; cocoindex.init()" 2>nul

REM Start backend in new window
echo Starting backend server...
start "CocoIndex Backend" cmd /k "python -m uvicorn app.main:app --port 8001 --log-level info"

REM Start Celery worker in new window
echo Starting Celery worker...
start "CocoIndex Celery" cmd /k "celery -A app.tasks worker --loglevel=info --concurrency=2"

REM Start frontend in new window
echo Starting frontend...
start "CocoIndex Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo.
echo All services started successfully!
echo.
echo Service URLs:
echo    Frontend:     http://localhost:3000
echo    Backend API:  http://localhost:8001
echo    Backend Docs: http://localhost:8001/docs
echo    Neo4j:        http://localhost:7474
echo    Qdrant:       http://localhost:6333/dashboard
echo.
echo Service Status:
docker-compose -f docker-compose.yml ps
echo.
echo Close this window to stop Docker services (application windows must be closed manually)
pause

REM Stop Docker services when user closes window
docker-compose -f docker-compose.yml down