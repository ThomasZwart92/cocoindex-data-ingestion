# CocoIndex Deployment Guide

## Quick Start (Localhost)

### Prerequisites
- Docker Desktop installed and running
- Python 3.11+ installed
- Node.js 18+ installed
- Git

### Step 1: Clone and Setup
```bash
git clone https://github.com/your-repo/cocoindex.git
cd cocoindex

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

### Step 2: Start Services

#### Windows:
```bash
start-local.bat
```

#### Mac/Linux:
```bash
chmod +x start-local.sh
./start-local.sh
```

This will:
1. Start all database services (Postgres, Redis, Neo4j, Qdrant)
2. Initialize CocoIndex
3. Start the backend API server
4. Start Celery workers for background tasks
5. Start the frontend development server

### Step 3: Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001
- **API Documentation**: http://localhost:8001/docs
- **Neo4j Browser**: http://localhost:7474 (neo4j/password123)
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Manual Setup

### 1. Start Infrastructure Services
```bash
docker-compose up -d
```

### 2. Initialize CocoIndex
```bash
python -c "import cocoindex; cocoindex.init()"
```

### 3. Start Backend
```bash
python -m uvicorn app.main:app --port 8001 --reload
```

### 4. Start Celery Worker
```bash
celery -A app.tasks worker --loglevel=info
```

### 5. Start Frontend
```bash
cd frontend
npm install
npm run dev
```

## Docker Deployment (Full Stack)

### Build and Run Everything
```bash
# Build images
docker-compose -f docker-compose.full.yml build

# Start all services
docker-compose -f docker-compose.full.yml up -d

# Check status
docker-compose -f docker-compose.full.yml ps

# View logs
docker-compose -f docker-compose.full.yml logs -f
```

### Stop Services
```bash
docker-compose -f docker-compose.full.yml down
```

## Production Deployment

### Cloud Platform Options

#### 1. AWS Deployment
- **Frontend**: S3 + CloudFront
- **Backend**: ECS Fargate or Elastic Beanstalk
- **Databases**: 
  - RDS PostgreSQL
  - ElastiCache Redis
  - Neptune (instead of Neo4j)
  - OpenSearch (instead of Qdrant)

#### 2. Google Cloud Platform
- **Frontend**: Cloud Storage + Cloud CDN
- **Backend**: Cloud Run or App Engine
- **Databases**:
  - Cloud SQL PostgreSQL
  - Memorystore Redis
  - Neo4j on Compute Engine
  - Vertex AI Vector Search

#### 3. Azure Deployment
- **Frontend**: Blob Storage + CDN
- **Backend**: Container Instances or App Service
- **Databases**:
  - Azure Database for PostgreSQL
  - Azure Cache for Redis
  - Neo4j on VMs
  - Azure Cognitive Search

### Environment Variables (Production)

Create a `.env.production` file:
```env
# API Keys (use secrets management in production)
OPENAI_API_KEY=<from-secret-manager>
GEMINI_API_KEY=<from-secret-manager>
SUPABASE_URL=<your-production-url>
SUPABASE_KEY=<from-secret-manager>

# Database URLs (use managed services)
DATABASE_URL=postgresql://user:pass@host:5432/cocoindex
REDIS_URL=redis://redis-cluster:6379
NEO4J_URI=bolt://neo4j-server:7687
QDRANT_URL=https://qdrant-cloud.example.com

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=warning
CORS_ORIGINS=https://yourdomain.com
```

### Security Checklist
- [ ] Use HTTPS everywhere
- [ ] Store secrets in secret management service
- [ ] Enable authentication on all databases
- [ ] Set up firewall rules
- [ ] Configure CORS properly
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure automated backups
- [ ] Implement health checks
- [ ] Set up CI/CD pipeline

## Monitoring

### Health Checks
- Backend: `GET http://localhost:8001/health`
- Postgres: `docker exec cocoindex-postgres pg_isready`
- Redis: `docker exec cocoindex-redis redis-cli ping`
- Neo4j: `curl http://localhost:7474`
- Qdrant: `curl http://localhost:6333/health`

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend
docker-compose logs -f celery-worker

# Backend logs (non-Docker)
tail -f server.log
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find and kill process using port
   netstat -ano | findstr :8001
   taskkill /PID <process-id> /F
   ```

2. **Docker services not starting**
   ```bash
   # Reset Docker data
   docker-compose down -v
   docker system prune -a
   ```

3. **Database connection errors**
   - Check if services are healthy: `docker-compose ps`
   - Verify connection strings in `.env`
   - Check firewall settings

4. **Celery not processing tasks**
   - Check Redis connection
   - Verify Celery is running: `celery -A app.tasks status`
   - Check Celery logs for errors

## Backup and Recovery

### Backup Data
```bash
# Postgres backup
docker exec cocoindex-postgres pg_dump -U postgres cocoindex > backup.sql

# Neo4j backup
docker exec cocoindex-neo4j neo4j-admin dump --to=/data/backup.dump

# Qdrant backup
curl -X POST http://localhost:6333/collections/documents/snapshots
```

### Restore Data
```bash
# Postgres restore
docker exec -i cocoindex-postgres psql -U postgres cocoindex < backup.sql

# Neo4j restore
docker exec cocoindex-neo4j neo4j-admin load --from=/data/backup.dump
```

## Support

For issues or questions:
1. Check the logs first
2. Consult CLAUDE.md for development context
3. Review the architecture documentation
4. Create an issue on GitHub