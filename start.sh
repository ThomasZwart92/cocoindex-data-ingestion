#!/bin/bash

# Simple start script for CocoIndex development
# Usage: ./start.sh [options]
#   Options:
#     --full       Start all services in Docker (default: infrastructure only)
#     --no-logs    Don't show logs after starting

set -e

FULL_MODE=false
SHOW_LOGS=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_MODE=true
            shift
            ;;
        --no-logs)
            SHOW_LOGS=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full] [--no-logs]"
            exit 1
            ;;
    esac
done

echo "🚀 Starting CocoIndex Development Environment..."

if [ "$FULL_MODE" = true ]; then
    echo "📦 Starting all services in Docker containers..."
    docker-compose -f docker-compose.full.yml up -d

    echo "⏳ Waiting for services to be healthy..."
    sleep 10

    echo "✅ All services started in Docker!"
    echo ""
    echo "Access points:"
    echo "  - Backend API: http://localhost:8005"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
else
    # Start infrastructure in Docker
    echo "🗄️ Starting infrastructure services..."
    docker-compose -f docker-compose.dev.yml up -d

    echo "⏳ Waiting for infrastructure to be healthy..."
    sleep 5

    # Check if virtual environment exists
    if [ -d ".venv312" ]; then
        PYTHON_CMD=".venv312/Scripts/python.exe"
    elif [ -d "venv312" ]; then
        PYTHON_CMD="venv312/Scripts/python.exe"
    elif [ -d ".venv" ]; then
        PYTHON_CMD=".venv/Scripts/python.exe"
    elif [ -d "venv" ]; then
        PYTHON_CMD="venv/Scripts/python.exe"
    else
        echo "❌ No virtual environment found. Please create one first."
        echo "   Run: python -m venv .venv312"
        exit 1
    fi

    # Start backend
    echo "🔧 Starting backend API..."
    $PYTHON_CMD -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8005 > logs/backend.log 2>&1 &
    echo $! > logs/backend.pid

    # Start Celery worker
    echo "👷 Starting Celery worker..."
    $PYTHON_CMD -m celery -A app.celery_app worker --loglevel=info --pool=solo > logs/celery.log 2>&1 &
    echo $! > logs/celery.pid

    # Start frontend
    echo "🎨 Starting frontend..."
    cd frontend
    npm run dev > ../logs/frontend.log 2>&1 &
    echo $! > ../logs/frontend.pid
    cd ..

    echo "✅ All services started!"
    echo ""
    echo "Access points:"
    echo "  - Backend API: http://localhost:8005"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Neo4j Browser: http://localhost:7474"
    echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
    echo ""
    echo "Logs:"
    echo "  - Backend: logs/backend.log"
    echo "  - Celery: logs/celery.log"
    echo "  - Frontend: logs/frontend.log"
fi

if [ "$SHOW_LOGS" = true ]; then
    echo ""
    echo "📋 Showing logs (Ctrl+C to stop viewing, services will continue running)..."
    if [ "$FULL_MODE" = true ]; then
        docker-compose -f docker-compose.full.yml logs -f
    else
        tail -f logs/*.log
    fi
fi