# CocoIndex Data Ingestion Portal - Continuation Prompt

## Project Overview
You're working on a CocoIndex Data Ingestion Portal that implements Anthropic's Contextual Retrieval approach for document processing with 67% improved retrieval accuracy. The project is ~45% complete.

## Current Session Focus: Implement Contextual Retrieval Pipeline

### Priority 1: Get NC2050 Document Processing with Contextual Chunks
The NC2050 document (ID: e86cbf9f-5fb7-4f5d-8ff9-44633a432a2d) has 1599 chars of content and needs to be processed through the contextual retrieval pipeline.

## Critical Files to Read First

### 1. Status & Architecture
- `IMPLEMENTATION_STATUS.md` - Project state, contextual retrieval approach
- `CLAUDE.md` - Critical rules (NEVER use mock data, always use CocoIndex)
- `architecture.md` - System design with contextual chunking details

### 2. Core Implementation Files
- `app/flows/document_processor.py` - CocoIndex flow (needs contextual chunking)
- `app/services/supabase_service.py` - Database service layer
- `app/connectors/notion_connector.py` - Notion connector (working)
- `contextual_retrieval_schema.sql` - SQL schema for contextual chunks

### 3. Test Files
- `test_simple_processing.py` - Simple processing test (partially working)

## Services Running
```bash
# Backend: FastAPI on port 8001
python -m uvicorn app.main:app --port 8001 --log-level info

# Frontend: Next.js on port 3000  
cd frontend && npm run dev

# Celery Worker
python -m celery -A app.worker worker --loglevel=info

# Docker Services
- Redis: Port 6379
- Qdrant: Port 6333
- Neo4j: Port 7687
```

## Environment Variables (.env)
```
SUPABASE_URL=https://ycddtddyffnqqehmjsid.supabase.co
SUPABASE_KEY=[set in .env]
NOTION_API_KEY_EMPLOYEE_ACCESS=[set in .env]
OPENAI_API_KEY=[set in .env]
ANTHROPIC_API_KEY=[set in .env]
COHERE_API_KEY=[set in .env if using reranking]
```

## Contextual Retrieval Implementation (Based on Anthropic Research)

### Architecture
```
Document
├── Page Chunks (1200 tokens)
│   ├── Paragraph Chunks (400 tokens)
│   │   ├── Original Text
│   │   ├── Contextual Summary (50-100 tokens via Claude Haiku)
│   │   └── Contextualized Text (summary + original)
│   └── Page Context
└── Document Context
```

### Expected Performance
- Contextual chunks alone: 35% reduction in retrieval failures
- + BM25 hybrid search: 49% reduction
- + Cohere reranking: 67% reduction

## Implementation Tasks (In Order)

### Task 1: Apply Contextual Retrieval Schema
```bash
# Run the schema updates
python execute_supabase_sql.py contextual_retrieval_schema.sql
```

### Task 2: Implement Contextual Chunking in CocoIndex Flow
Update `app/flows/document_processor.py` to:
1. Create hierarchical chunks (page → paragraph)
2. Generate contextual summaries using Claude Haiku
3. Store both original and contextualized text
4. Generate embeddings on contextualized text
5. Create BM25 tokens for lexical search

### Task 3: Process NC2050 Document
```python
from app.flows.document_processor import create_and_run_flow
document_id = "e86cbf9f-5fb7-4f5d-8ff9-44633a432a2d"  # NC2050
result = create_and_run_flow(document_id)
```

### Task 4: Implement Hybrid Search
Create `app/services/hybrid_search.py`:
1. Semantic search via Qdrant (embeddings)
2. BM25 search on contextualized text
3. Reciprocal rank fusion
4. Optional: Cohere reranking

### Task 5: Verify Results
- Check chunks table has contextual_summary populated
- Verify hierarchical relationships (parent_chunk_id)
- Test search with complex query
- Measure retrieval accuracy

## Key Implementation Notes

### CocoIndex Flow Pattern
```python
@cocoindex.flow_def(name="DocumentProcessor")
def process_document_flow(flow_builder, data_scope):
    # Use declarative transformations
    with data_scope["doc"].row() as doc:
        # Create hierarchical chunks
        doc["page_chunks"] = doc["content"].transform(...)
        
        with doc["page_chunks"].row() as page:
            page["para_chunks"] = page["text"].transform(...)
            
            with page["para_chunks"].row() as chunk:
                # Generate context
                chunk["context"] = chunk["text"].transform(
                    GenerateContext(doc_title, page_context)
                )
                # Contextualize
                chunk["contextualized"] = combine(
                    chunk["context"], 
                    chunk["text"]
                )
```

### Critical Rules
1. **NEVER use mock data** - Always real API integrations
2. **ALWAYS use CocoIndex dataflow** - No imperative processing
3. **Track document state** - Use state machine
4. **Table names**: Use `chunks` and `entities` (not `document_chunks`)

## Current Issues Resolved
- ✅ Notion content fetching (fixed, 1599 chars for NC2050)
- ✅ Table naming (using `chunks` not `document_chunks`)
- ⚠️ CocoIndex flow needs contextual chunking implementation
- ⚠️ Celery worker has some errors but is running

## Success Metrics
- [ ] NC2050 has hierarchical chunks with context
- [ ] Contextual summaries are 50-100 tokens
- [ ] BM25 tokens stored for lexical search
- [ ] Embeddings generated on contextualized text
- [ ] Search returns relevant results

## Testing Commands
```python
# Check document content
from app.services.supabase_service import SupabaseService
service = SupabaseService()
docs = service.list_documents()
nc2050 = [d for d in docs if 'NC2050' in d.name][0]
print(f"Content: {nc2050.content[:500]}")

# Process document
from app.flows.document_processor import create_and_run_flow
result = create_and_run_flow("e86cbf9f-5fb7-4f5d-8ff9-44633a432a2d")

# Check chunks
chunks = service.client.table("chunks").select("*").eq(
    "document_id", "e86cbf9f-5fb7-4f5d-8ff9-44633a432a2d"
).execute()
print(f"Found {len(chunks.data)} chunks")
```

## References
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [CocoIndex Docs](https://cocoindex.io/docs)

The goal is to implement the contextual retrieval pipeline and successfully process the NC2050 document with hierarchical, context-enriched chunks for superior search accuracy.