# Task Tracker - Data Ingestion Portal

## 🔴 CRITICAL ISSUES - FIX IMMEDIATELY (Data Integrity at Risk)

### Database Transaction Management
- [ ] **Implement explicit transaction boundaries** - Currently using autocommit=False but no commit/rollback calls
  - [ ] Wrap all multi-step database operations in try/except/finally blocks
  - [ ] Add db.session.commit() on success
  - [ ] Add db.session.rollback() on failure
  - [ ] Priority files: app/api/documents.py, app/api/chunks.py, app/api/entities.py
- [ ] **Add connection pooling** - Currently creating new connections for each request
  - [ ] Configure SQLAlchemy connection pool in app/services/database.py
  - [ ] Set pool_size and max_overflow parameters

### External API Resilience  
- [ ] **Apply retry logic to ALL external services** - Only LLM service has retry
  - [ ] Copy retry decorator from app/services/llm_service.py
  - [ ] Apply to app/connectors/notion_connector.py
  - [ ] Apply to app/connectors/google_drive_connector.py
  - [ ] Apply to app/services/embedding_service.py
- [ ] **Add client-side rate limiting** - Risk of API ban from Notion/Google
  - [ ] Implement rate limiting for Notion API calls
  - [ ] Implement rate limiting for Google Drive API calls

## 🟡 HIGH PRIORITY - Fix Before Production

### Error Handling & Observability
- [ ] **Replace generic exception handling** - Currently catching Exception everywhere
  - [ ] Replace with specific exceptions (SQLAlchemyError, RequestException, etc.)
  - [ ] Add proper logging before re-raising
  - [ ] Don't expose internal errors to users
- [ ] **Add React Error Boundaries** - Frontend crashes on errors
  - [ ] Add top-level error boundary in frontend/app/layout.tsx
  - [ ] Add error boundaries around major components
- [ ] **Add loading states** - No feedback during async operations
  - [ ] Add loading spinners for all API calls
  - [ ] Add skeleton loaders for data fetching

### Input Validation & Security
- [ ] **Add input validation** - User input goes directly to database
  - [ ] Use Pydantic models for all API endpoints
  - [ ] Validate all query parameters
  - [ ] Sanitize JSON inputs
- [ ] **Remove fake auth code** - Misleading security theater
  - [ ] Replace app/services/auth_service.py with NotImplementedError
  - [ ] Remove token logic from frontend until real auth is ready
  - [ ] Add clear TODO comments about auth being deferred

## 🎯 Immediate Tasks (High Value)

### Phase 2: API Development
- [ ] **Document Management Endpoints**
  - [ ] GET /api/documents (list all)
  - [ ] GET /api/documents/{id} (with chunks)
  - [ ] DELETE /api/documents/{id}
  - [ ] POST /api/documents/{id}/reprocess
  
- [ ] **Processing Endpoints**
  - [ ] POST /api/process/notion
  - [ ] POST /api/process/gdrive
  - [ ] GET /api/jobs/{id}/status
  
- [ ] **Chunk Management**
  - [ ] PUT /api/chunks/{id} (edit chunk text)
  - [ ] POST /api/documents/{id}/rechunk
  
- [ ] **Entity Management**
  - [ ] GET /api/documents/{id}/entities
  - [ ] PUT /api/entities/{id} (edit entity)
  - [ ] DELETE /api/entities/{id}
  - [ ] POST /api/entities (add new)

### Phase 2: Basic UI
- [ ] **Next.js Frontend Setup**
  - [ ] Basic layout with navigation
  - [ ] Authentication integration
  
- [ ] **Document List Page**
  - [ ] Table with status indicators
  - [ ] Processing state badges
  - [ ] Delete action
  
- [ ] **Document Detail Page**
  - [ ] Metadata display and edit form
  - [ ] Chunk list with edit capability
  - [ ] Entity list with edit/delete
  - [ ] Reprocess button
  
- [ ] **Processing Status**
  - [ ] Job progress indicator
  - [ ] Error message display
  - [ ] Retry failed jobs

### Phase 3: Quality Control Features
- [ ] **Review Queue**
  - [ ] Queue page for pending documents
  - [ ] Preview all extracted data before approval
  - [ ] Approve/Reject with comments
  - [ ] Partial approval (approve chunks, reject entities)
  
- [ ] **Chunking UI**
  - [ ] Chunk context visualization (before/after text)
  - [ ] Chunk size indicator
  - [ ] Overlap visualization
  
- [ ] **Multi-Model Comparison**
  - [ ] Display GPT-4 suggestions
  - [ ] Display Gemini suggestions
  - [ ] Highlight differences
  - [ ] Manual selection interface

### Documentation Tasks
- [ ] API documentation with examples
- [ ] User guide for review process
- [ ] Source setup guide (Notion & Google Drive)
- [ ] Relationship types guide with examples

## 🔧 Technical Debt (Medium Priority)

### Code Quality
- [ ] Replace generic exception handling with specific exceptions
- [ ] Add connection pooling for database clients
- [ ] Add input validation for all public methods
- [ ] Implement caching for expensive operations
- [ ] Add comprehensive logging throughout

### Pipeline Enhancements
- [ ] Connect source connectors to unified pipeline
- [ ] Implement source scheduling (periodic scans)
- [ ] Add source-specific metadata extraction
- [ ] Create unified processing queue
- [ ] Add confidence thresholds for relationship extraction

### Knowledge Graph Improvements
- [ ] Implement relationship validation rules
- [ ] Create graph visualization endpoint
- [ ] Add relationship editing API
- [ ] Improve entity resolution algorithms

## 🔒 Security & Scalability Tasks (Lower Priority)

### Security Hardening
- [ ] Implement authentication middleware for API endpoints
- [ ] Add rate limiting for API endpoints
- [ ] Implement API rate limit handling for external sources
- [ ] Add audit logging for all data modifications
- [ ] Implement field-level encryption for sensitive data

### Database & Performance
- [ ] Implement database transactions for atomic operations
- [ ] Fix N+1 query problem in chunk updates
- [ ] Implement streaming for large file processing (prevent OOM)
- [ ] Add database connection pooling
- [ ] Implement dependency injection pattern

### Monitoring & Observability
- [ ] Set up error tracking (Sentry or similar)
- [ ] Create job status dashboard
- [ ] Add processing metrics collection
- [ ] Implement API endpoint monitoring
- [ ] Add database connection health checks

### Deployment (Phase 4)
- [ ] Configure Railway environment variables
- [ ] Deploy backend service
- [ ] Deploy frontend service
- [ ] Deploy Celery worker
- [ ] Set up domain and SSL

## 🚀 Future Enhancements (Post-MVP)

### Advanced Features
- [ ] **Image Processing Pipeline**
  - [ ] Extract images from PDFs using pdf2image
  - [ ] GPT-4 Vision integration for captioning
  - [ ] ColPali visual embeddings
  - [ ] Store images in Supabase storage
  
- [ ] **Document Intelligence**
  - [ ] Custom entity types
  - [ ] Domain-specific extractors
  - [ ] Document versioning
  - [ ] Change tracking and diff view
  
- [ ] **Scale Features** (When > 500 docs)
  - [ ] Batch operations UI
  - [ ] Auto-approval based on confidence
  - [ ] WebSocket real-time updates
  - [ ] Parallel processing optimization

## ✅ Completed Tasks

### Phase 0: Foundation
- [x] Set up Redis (local)
- [x] Configure Celery with Redis broker
- [x] Set up Supabase project
- [x] Configure Qdrant (Docker)
- [x] Set up Neo4j (Docker)
- [x] Implement document state machine
- [x] Create database schema with state tracking

### Phase 1: Core Pipeline
- [x] **Source Connectors**
  - [x] Notion connector with API authentication
  - [x] Google Drive connector with service account
  - [x] Change detection for incremental updates
  
- [x] **Security Model**
  - [x] 5-tier security hierarchy
  - [x] Automatic security tagging
  - [x] Security metadata on all documents
  
- [x] **Knowledge Graph**
  - [x] 14 relationship types implementation
  - [x] Entity resolution logic
  - [x] Hybrid extraction (rules + LLM)
  
- [x] **Search Implementation**
  - [x] Vector search endpoint (<200ms)
  - [x] Graph search
  - [x] Hybrid search
  
- [x] **LLM Integration**
  - [x] OpenAI API client
  - [x] Gemini API client
  - [x] Multi-model comparison
  
- [x] **Testing**
  - [x] Notion integration test (NC2068)
  - [x] Google Drive test (Refill manual)
  - [x] Search latency validation

## 📊 Task Priorities

1. **High Value, Low Effort**: API endpoints, Basic UI
2. **High Value, Medium Effort**: Review queue, Quality control
3. **Medium Value, Low Effort**: Documentation, Code quality
4. **Low Value (for MVP)**: Security hardening, Scalability