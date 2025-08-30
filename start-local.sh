#!/bin/bash

# CocoIndex Local Deployment Script
# This script starts all services locally using Docker Compose

set -e

echo "🥥 Starting CocoIndex Local Deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your API keys"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running!"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Stop any existing containers
echo "📦 Stopping existing containers..."
docker-compose -f docker-compose.yml down 2>/dev/null || true
docker-compose -f docker-compose.full.yml down 2>/dev/null || true

# Start infrastructure services first
echo "🚀 Starting infrastructure services..."
docker-compose -f docker-compose.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose -f docker-compose.yml ps

# Initialize CocoIndex
echo "🔧 Initializing CocoIndex..."
python -c "import cocoindex; cocoindex.init()" 2>/dev/null || true

# Start backend
echo "🎯 Starting backend server..."
python -m uvicorn app.main:app --port 8001 --log-level info &
BACKEND_PID=$!

# Start Celery worker
echo "⚙️ Starting Celery worker..."
celery -A app.tasks worker --loglevel=info --concurrency=2 &
CELERY_PID=$!

# Start frontend
echo "🎨 Starting frontend..."
cd frontend
npm install
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ All services started successfully!"
echo ""
echo "📌 Service URLs:"
echo "   Frontend:    http://localhost:3000"
echo "   Backend API: http://localhost:8001"
echo "   Backend Docs: http://localhost:8001/docs"
echo "   Neo4j:       http://localhost:7474"
echo "   Qdrant:      http://localhost:6333/dashboard"
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.yml ps
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for user interrupt
trap "echo '🛑 Stopping services...'; kill $BACKEND_PID $CELERY_PID $FRONTEND_PID 2>/dev/null; docker-compose -f docker-compose.yml down; exit" INT
wait