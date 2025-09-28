# Suggested Commands for Development

## Starting the Application

### Local Development with Docker
```bash
# Start infrastructure (PostgreSQL, Redis, Neo4j, Qdrant)
docker-compose up -d

# Start backend API server
python -m uvicorn app.main:app --reload --port 8001

# Start Celery worker for background tasks
python -m celery -A app.worker worker --loglevel=info

# Start frontend development server
cd frontend && npm run dev
```

### Quick Start Script
```bash
# Windows
start-local.bat

# Unix/Mac
./start-local.sh
```

## Testing Commands
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_three_tier_chunking.py

# Run with coverage
pytest --cov=app tests/

# Run only unit tests
pytest tests/ -k "not integration"
```

## Database Commands
```bash
# Check Supabase connection
python test_db_connection.py

# Setup database tables
python setup_supabase_tables.py

# Clear test documents
python clear_test_documents.py
```

## Document Processing
```bash
# Process a document with three-tier chunking
python process_document_three_tier.py

# Test contextual processing
python test_contextual_processing.py

# Extract NC2050 relationships
python extract_nc2050_relationships.py
```

## Utility Commands
```bash
# Check environment variables
python -c "from app.config import settings; print(settings)"

# Validate configuration
python app/config_validator.py

# Check API endpoints
curl http://localhost:8001/api/documents
curl http://localhost:8001/api/health

# View logs
tail -f server.log
tail -f server_8001.log
```

## Git Commands (Windows-specific)
```bash
# Status
git status

# Add and commit
git add .
git commit -m "message"

# Push changes
git push origin main
```

## Frontend Commands
```bash
cd frontend

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Run production build
npm start
```

## Docker Management
```bash
# View running containers
docker ps

# View logs
docker logs cocoindex-postgres
docker logs cocoindex-redis
docker logs cocoindex-neo4j
docker logs cocoindex-qdrant

# Stop all services
docker-compose down

# Reset with volume cleanup
docker-compose down -v
```