# 🚨 URGENT TASKS - CocoIndex Data Ingestion Portal

## Critical Architecture Pivot Required
**Current State**: System built with imperative service calls  
**Required State**: Declarative CocoIndex dataflow pattern  
**Deadline**: 2 weeks to functional MVP

---

## 🔴 WEEK 1: Core Pipeline Refactor (Days 1-5)

### Day 1-2: Implement Proper CocoIndex Flow
**Goal**: Replace imperative processing with declarative dataflow

#### Tasks:
- [ ] **Fix document_ingestion_flow.py to use proper CocoIndex pattern**
  ```python
  # Location: app/flows/document_ingestion_flow.py
  # Replace entire flow with proper implementation
  ```
  
- [ ] **Create CocoIndex function wrappers for existing services**
  ```python
  # Create: app/flows/functions/llm_extraction.py
  @cocoindex.op.function()
  def extract_entities_with_llm(text: str) -> List[Entity]:
      # Wrap existing LLMService.extract_entities
  ```

- [ ] **Implement row context pattern**
  ```python
  with data_scope["documents"].row() as doc:
      doc["chunks"] = doc["content"].transform(...)
      doc["entities"] = doc["content"].transform(...)
  ```

#### Test Script:
```bash
# Create: test_cocoindex_flow.py
python test_cocoindex_flow.py
# Should process a sample .md file through the pipeline
# Verify chunks appear in Qdrant
# Verify entities appear in Neo4j
```

### Day 2-3: Document State Machine Implementation
**Goal**: Track document lifecycle from discovery to ingestion

#### Tasks:
- [ ] **Create state machine model**
  ```python
  # Create: app/models/document_state.py
  class DocumentState(Enum):
      DISCOVERED = "discovered"
      PROCESSING = "processing"
      PENDING_REVIEW = "pending_review"
      APPROVED = "approved"
      INGESTED = "ingested"
      FAILED = "failed"
  ```

- [ ] **Use existing Supabase tables (already set up)**
  ```python
  # NOTE: Supabase is our PostgreSQL database
  # Connection info in .env:
  # - SUPABASE_URL
  # - SUPABASE_KEY
  # - DATABASE_URL (PostgreSQL connection string)
  
  # Tables already created:
  # - documents
  # - document_chunks
  # - ingestion_queue
  # - source_configs
  # (See setup_supabase_tables.py for schema)
  ```

- [ ] **Add additional state tracking tables if needed**
  ```sql
  -- Only if not already in Supabase schema
  -- Check existing tables first with:
  -- python execute_supabase_sql.py --list-tables
  ```

- [ ] **Create state transition service**
  ```python
  # Create: app/services/state_manager.py
  class DocumentStateManager:
      async def transition(document_id: str, new_state: DocumentState)
      async def can_transition(document_id: str, new_state: DocumentState) -> bool
      async def get_current_state(document_id: str) -> DocumentState
  ```

#### Test Script:
```bash
# Create: test_state_machine.py
python test_state_machine.py
# Should test all valid state transitions
# Should reject invalid transitions
# Should handle concurrent state changes
```

### Day 3-4: Wire Up End-to-End Pipeline
**Goal**: Connect API → Celery → CocoIndex → Databases

#### Tasks:
- [ ] **Fix Celery worker to trigger real CocoIndex flows**
  ```python
  # Update: app/worker.py
  @celery_app.task(bind=True, max_retries=3)
  def process_document(self, document_id: str):
      # 1. Update state to PROCESSING
      # 2. Run CocoIndex flow
      # 3. Update state based on result
      flow = document_ingestion_flow()
      result = flow.run(document_id=document_id)
  ```

- [ ] **Create document ingestion API endpoint**
  ```python
  # Update: app/api/documents.py
  @router.post("/ingest")
  async def ingest_document(
      source_type: str,
      source_id: str,
      priority: int = 0
  ):
      # 1. Create document record
      # 2. Set state to DISCOVERED
      # 3. Queue for processing
      task = process_document.delay(document_id)
  ```

- [ ] **Connect source connectors to pipeline**
  ```python
  # Create: app/services/source_scanner.py
  class SourceScanner:
      async def scan_notion(self):
          # Use NotionConnector
          # For each new/changed document:
          #   - Call ingest API
      
      async def scan_gdrive(self):
          # Use GoogleDriveConnector
          # For each new/changed document:
          #   - Call ingest API
  ```

#### Test Script:
```bash
# Create: test_end_to_end.py
python test_end_to_end.py
# Should ingest from Notion
# Should track state changes
# Should store in Qdrant + Neo4j
# Should handle failures gracefully
```

### Day 5: LlamaParse Integration
**Goal**: Add proper PDF parsing with quality tiers

#### Tasks:
- [ ] **Install and configure LlamaParse**
  ```bash
  pip install llama-parse
  # Add LLAMA_PARSE_API_KEY to .env
  ```

- [ ] **Create parsing service with tier strategy**
  ```python
  # Create: app/services/document_parser.py
  class DocumentParser:
      async def parse_pdf(
          self,
          content: bytes,
          tier: str = "balanced"  # balanced, agentic, agentic_plus
      ) -> tuple[str, dict]:
          parser = LlamaParse(
              api_key=settings.llama_parse_api_key,
              parsing_mode=tier,
              result_type="markdown"
          )
          result = await parser.parse_raw(content)
          return result.text, {"tier": tier, "confidence": result.confidence}
  ```

- [ ] **Add parsing to CocoIndex flow**
  ```python
  # Update: app/flows/document_ingestion_flow.py
  @cocoindex.op.function()
  def parse_document(content: bytes) -> str:
      parser = DocumentParser()
      text, metadata = parser.parse_pdf(content)
      return text
  ```

#### Test Script:
```bash
# Create: test_llamaparse.py
python test_llamaparse.py
# Should parse a sample PDF
# Should extract text and tables
# Should track parsing costs
```

---

## 🟠 WEEK 1-2: Quality Control Backend (Days 6-8)

### Day 6: Review & Approval APIs
**Goal**: Enable human-in-the-loop quality control

#### Tasks:
- [ ] **Create chunk review endpoints**
  ```python
  # Update: app/api/chunks.py
  @router.get("/documents/{id}/chunks")
  async def get_chunks_for_review(document_id: str)
  
  @router.put("/chunks/{id}")
  async def update_chunk(chunk_id: str, text: str)
  
  @router.post("/chunks/{id}/approve")
  async def approve_chunk(chunk_id: str)
  ```

- [ ] **Create entity correction endpoints**
  ```python
  # Update: app/api/entities.py
  @router.get("/documents/{id}/entities")
  async def get_entities_for_review(document_id: str)
  
  @router.put("/entities/{id}")
  async def update_entity(entity_id: str, data: dict)
  
  @router.post("/entities/merge")
  async def merge_entities(entity_ids: List[str])
  ```

- [ ] **Create approval workflow endpoints**
  ```python
  # Update: app/api/approval.py
  @router.get("/queue/pending")
  async def get_pending_approvals()
  
  @router.post("/documents/{id}/approve")
  async def approve_document(document_id: str)
  
  @router.post("/documents/{id}/reject")
  async def reject_document(document_id: str, reason: str)
  ```

#### Test Script:
```bash
# Create: test_approval_workflow.py
python test_approval_workflow.py
# Should get pending documents
# Should allow chunk editing
# Should allow entity correction
# Should transition states correctly
```

### Day 7: Multi-Model Comparison
**Goal**: Enable comparison between GPT-4 and Gemini outputs

#### Tasks:
- [ ] **Create comparison storage schema**
  ```sql
  -- Create: migrations/002_model_comparisons.sql
  CREATE TABLE model_comparisons (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      document_id UUID NOT NULL,
      comparison_type VARCHAR(50), -- 'entities', 'metadata'
      gpt4_output JSONB,
      gemini_output JSONB,
      selected_output VARCHAR(50),
      user_modifications JSONB,
      created_at TIMESTAMPTZ DEFAULT NOW()
  );
  ```

- [ ] **Update LLM extraction to store comparisons**
  ```python
  # Update: app/services/llm_service.py
  async def extract_with_comparison(self, text: str):
      gpt4_result = await self._call_openai(...)
      gemini_result = await self._call_gemini(...)
      
      # Store comparison
      await self.store_comparison(
          gpt4=gpt4_result,
          gemini=gemini_result
      )
      
      return {"gpt4": gpt4_result, "gemini": gemini_result}
  ```

#### Test Script:
```bash
# Create: test_model_comparison.py
python test_model_comparison.py
# Should extract with both models
# Should store comparisons
# Should allow selection
```

### Day 8: Cost Tracking & Monitoring
**Goal**: Track and control expensive operations

#### Tasks:
- [ ] **Create cost tracking table**
  ```sql
  -- Create: migrations/003_cost_tracking.sql
  CREATE TABLE api_costs (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      document_id UUID,
      service VARCHAR(50), -- 'openai', 'gemini', 'llamaparse'
      operation VARCHAR(100),
      tokens_used INTEGER,
      cost_usd DECIMAL(10, 6),
      metadata JSONB,
      created_at TIMESTAMPTZ DEFAULT NOW()
  );
  ```

- [ ] **Add cost tracking decorators**
  ```python
  # Create: app/utils/cost_tracking.py
  def track_cost(service: str, operation: str):
      def decorator(func):
          async def wrapper(*args, **kwargs):
              start_tokens = get_token_count()
              result = await func(*args, **kwargs)
              tokens_used = get_token_count() - start_tokens
              cost = calculate_cost(service, tokens_used)
              await log_cost(service, operation, tokens_used, cost)
              return result
          return wrapper
      return decorator
  ```

- [ ] **Add rate limiting**
  ```python
  # Create: app/utils/rate_limiting.py
  class RateLimiter:
      def __init__(self, max_per_minute: int):
          self.max_per_minute = max_per_minute
          self.calls = deque()
      
      async def check_limit(self) -> bool:
          # Return True if under limit
  ```

#### Test Script:
```bash
# Create: test_cost_tracking.py
python test_cost_tracking.py
# Should track LLM costs
# Should track LlamaParse costs
# Should enforce rate limits
# Should generate cost reports
```

---

## 🟡 WEEK 2: Advanced Features & UI (Days 9-12)

### Day 9-10: Image Processing Pipeline
**Goal**: Extract and process images from documents

#### Tasks:
- [ ] **Implement image extraction**
  ```python
  # Create: app/services/image_processor.py
  class ImageProcessor:
      async def extract_images_from_pdf(self, pdf_bytes: bytes):
          # Use pdf2image
          
      async def generate_caption(self, image_bytes: bytes):
          # Use GPT-4 Vision
          
      async def generate_embeddings(self, image_bytes: bytes):
          # Use ColPali
  ```

- [ ] **Add image storage to Supabase**
  ```python
  # Update: app/services/supabase_service.py
  async def store_image(self, image_bytes: bytes, metadata: dict):
      # Store in Supabase storage
      # Return URL
  ```

#### Test Script:
```bash
# Create: test_image_processing.py
python test_image_processing.py
# Should extract images from PDF
# Should generate captions
# Should store in Supabase
# Should create visual embeddings
```

### Day 11-12: Basic Review UI
**Goal**: Minimal interface for document review

#### Tasks:
- [ ] **Create review dashboard page**
  ```typescript
  // Create: frontend/app/review/page.tsx
  // List pending documents
  // Show processing status
  // Allow navigation to detail view
  ```

- [ ] **Create chunk editing interface**
  ```typescript
  // Create: frontend/app/review/[id]/chunks/page.tsx
  // Display chunks with context
  // Allow inline editing
  // Show before/after preview
  ```

- [ ] **Create entity correction interface**
  ```typescript
  // Create: frontend/app/review/[id]/entities/page.tsx
  // Display extracted entities
  // Allow editing/merging
  // Show confidence scores
  ```

#### Test Script:
```bash
# Create: test_ui_integration.py
python test_ui_integration.py
# Should load review dashboard
# Should allow chunk editing
# Should save changes via API
```

---

## 📊 Testing & Validation Plan

### Daily Test Suite
```bash
#!/bin/bash
# Create: run_daily_tests.sh

echo "Running CocoIndex Flow Tests..."
python test_cocoindex_flow.py

echo "Running State Machine Tests..."
python test_state_machine.py

echo "Running End-to-End Tests..."
python test_end_to_end.py

echo "Running API Tests..."
pytest app/tests/test_api.py

echo "Checking Performance..."
python test_search_latency.py
```

### Integration Test Checklist
- [ ] Process document from Notion → Qdrant + Neo4j
- [ ] Process PDF from Google Drive → Parse → Store
- [ ] Edit chunk → Verify update in Qdrant
- [ ] Correct entity → Verify update in Neo4j
- [ ] Reject document → Verify state transition
- [ ] Retry failed document → Verify reprocessing

### Performance Validation
- [ ] Vector search < 200ms
- [ ] Document processing < 30s
- [ ] Chunk editing < 1s response
- [ ] State transitions < 500ms

---

## 🎯 Success Metrics

### Week 1 Goals
- [ ] CocoIndex flow processing documents end-to-end
- [ ] State machine tracking all documents
- [ ] LlamaParse parsing PDFs successfully
- [ ] Source connectors triggering ingestion

### Week 2 Goals
- [ ] Review APIs fully functional
- [ ] Cost tracking operational
- [ ] Basic UI for chunk review
- [ ] All tests passing

### MVP Completion Criteria
- [ ] Can ingest from Notion and Google Drive
- [ ] Can parse PDFs with LlamaParse
- [ ] Can review and edit chunks before storage
- [ ] Can correct entities before graph storage
- [ ] Tracks all costs and state transitions
- [ ] Meets <200ms search latency requirement

---

## 🚀 Quick Start Commands

```bash
# Start all services
docker-compose up -d
python -m celery -A app.worker worker --loglevel=info &
python -m uvicorn app.main:app --reload &
cd frontend && npm run dev &

# Run test suite
./run_daily_tests.sh

# Check system status
curl http://localhost:8000/health
curl http://localhost:3000

# Monitor logs
tail -f logs/celery.log
tail -f logs/fastapi.log
```

---

## ⚠️ Critical Dependencies

1. **Environment Variables Required (check .env file)**:
   - `SUPABASE_URL` - Supabase project URL
   - `SUPABASE_KEY` - Supabase anon/service key
   - `DATABASE_URL` - PostgreSQL connection string (Supabase)
   - `OPENAI_API_KEY` - For GPT-4 and embeddings
   - `GOOGLE_AI_API_KEY` - For Gemini
   - `LLAMA_PARSE_API_KEY` - Get from LlamaParse
   - `NOTION_API_KEY_*` - Notion tokens per security level
   - `QDRANT_URL` - Vector database URL
   - `QDRANT_API_KEY` - Qdrant API key
   - `NEO4J_URI` - Graph database URI
   - `NEO4J_USERNAME` - Neo4j username
   - `NEO4J_PASSWORD` - Neo4j password
   - `REDIS_URL` - Redis connection URL

2. **Services Must Be Running**:
   - Supabase (PostgreSQL) - Already configured in .env
   - Redis (port 6379) - via Docker
   - Qdrant (port 6333) - via Docker
   - Neo4j (ports 7474, 7687) - via Docker
   
3. **CocoIndex State Database**:
   - CocoIndex uses PostgreSQL for incremental processing state
   - We'll use a separate schema in Supabase for this
   - Connection configured via COCOINDEX_DATABASE_URL

3. **Python Dependencies**:
   ```bash
   pip install llama-parse pdf2image cocoindex
   ```

---

## 📝 Notes for Developers

1. **NEVER skip the CocoIndex pattern** - All processing must use dataflow
2. **ALWAYS update state** - Every operation must track document state
3. **TEST after each component** - Don't build on broken foundations
4. **TRACK costs** - Every LLM call must be logged
5. **HANDLE failures** - Every operation needs retry logic

---

## 🔄 Daily Standup Questions

1. Is the CocoIndex flow processing documents?
2. Are state transitions being tracked?
3. Are all tests passing?
4. What's blocking progress?
5. What help is needed?

---

**Last Updated**: 2025-08-20
**Priority**: CRITICAL - System non-functional without these tasks
**Estimated Completion**: 2 weeks with focused effort