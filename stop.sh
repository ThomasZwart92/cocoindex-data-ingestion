#!/bin/bash

# Simple stop script for CocoIndex development
# Usage: ./stop.sh [options]
#   Options:
#     --full        Stop Docker containers too (default: only stops host services)
#     --clean       Remove volumes and clean data

set -e

STOP_CONTAINERS=false
CLEAN_DATA=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            STOP_CONTAINERS=true
            shift
            ;;
        --clean)
            CLEAN_DATA=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full] [--clean]"
            exit 1
            ;;
    esac
done

echo "🛑 Stopping CocoIndex services..."

# Stop host-based services if PIDs exist
if [ -d "logs" ]; then
    for pidfile in logs/*.pid; do
        if [ -f "$pidfile" ]; then
            SERVICE=$(basename "$pidfile" .pid)
            PID=$(cat "$pidfile")
            if kill -0 "$PID" 2>/dev/null; then
                echo "  Stopping $SERVICE (PID: $PID)..."
                kill "$PID" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
fi

# Stop any orphaned processes on known ports
for PORT in 8005 8001 3000; do
    PID=$(lsof -ti :$PORT 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "  Stopping process on port $PORT (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
    fi
done

if [ "$STOP_CONTAINERS" = true ] || [ "$CLEAN_DATA" = true ]; then
    echo "🐳 Stopping Docker containers..."

    # Try both compose files to ensure we stop everything
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    docker-compose -f docker-compose.full.yml down 2>/dev/null || true
    docker-compose down 2>/dev/null || true

    if [ "$CLEAN_DATA" = true ]; then
        echo "🗑️ Cleaning up volumes and data..."
        docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
        docker-compose -f docker-compose.full.yml down -v 2>/dev/null || true
        rm -rf logs/*.log logs/*.pid
        echo "  All data cleaned!"
    fi
fi

echo "✅ All services stopped!"