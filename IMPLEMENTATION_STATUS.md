# CocoIndex Data Ingestion Portal - Implementation Status

## Current State Overview
**Date:** December 24, 2024  
**Overall Progress:** ~45% Complete - Infrastructure ready, contextual retrieval approach defined
**Latest Update:** Adopted Anthropic's Contextual Retrieval approach for 67% reduction in retrieval failures

## 🎯 NEW: Contextual Retrieval Implementation
Based on [Anthropic's research](https://www.anthropic.com/news/contextual-retrieval), we're implementing:
- **Hierarchical chunking**: Page (1200) → Paragraph (400) → Semantic chunks
- **Contextual enrichment**: 50-100 token summaries prepended to each chunk
- **Hybrid search**: BM25 + semantic embeddings with reciprocal rank fusion
- **Reranking**: Cohere reranker to select top 20 from 150 candidates
- **Expected improvement**: 67% reduction in retrieval failures

## 🚨 ARCHITECTURAL GAP - Not Using CocoIndex

### The Problem
This entire project was built to leverage CocoIndex for data transformation, but the current implementation **completely bypasses CocoIndex** and manually reimplements (poorly) what CocoIndex provides out-of-the-box:

**Missing CocoIndex Benefits:**
- ❌ **No Incremental Processing**: Reprocesses everything on each run
- ❌ **No Data Lineage**: Can't track transformations or debug issues
- ❌ **No Automatic Retries**: Manual error handling only
- ❌ **No Multi-DB Sync**: Manual saves to each database separately
- ❌ **No Built-in Functions**: Using basic text splitting instead of language-aware chunking
- ❌ **No Caching**: Paying for same LLM calls repeatedly

### The Solution: Hybrid CocoIndex Approach
Wrap existing processing in CocoIndex flows while keeping current API structure. This gives us CocoIndex benefits without frontend changes.

## ✅ COMPLETED COMPONENTS

### 1. Infrastructure & Database Layer
- **Supabase Setup:** Connected and operational with 8 tables
  - `documents` table with content column ✅
  - `chunks`, `entities`, `relationships` tables (schema exists) ✅
  - Table naming fixed (was using wrong names) ✅
  - `jobs`, `ingestion_queue`, `source_configs` tables
- **Qdrant Vector DB:** Docker container running on port 6333
- **Neo4j Graph DB:** Docker container running on port 7687
- **Redis + Celery:** Async processing infrastructure ready
- **FastAPI Backend:** Running on port 8001
- **Next.js Frontend:** Running on port 3000

### 2. Document Processing Pipeline (Partially Working)
- **LlamaParse Integration:** Document parser configured
- **Chunking System:** Basic DocumentChunker implemented (not using CocoIndex)
- **Entity Extraction:** Basic EntityExtractor ready (not using CocoIndex)
- **State Machine:** Document lifecycle (discovered → processing → pending_review)

### 3. Source Connectors (Need Fix)
- **Notion Connector:** Code exists but API token invalid
- **Google Drive Connector:** Service account authentication configured
- **Content Fetching:** ❌ NOT WORKING - documents have no content

### 4. Frontend Features
- **Document List View:** Shows all documents with status indicators
- **Processing UI:** Spinner animations and auto-refresh for processing documents
- **Document Detail Page:** Routes working, displays document metadata
- **Paper Craft Design:** Consistent styling applied throughout

## 🔴 CRITICAL GAPS - System Breaking Issues

### 1. ~~Database Table Naming Mismatch~~ ✅ FIXED
- Already resolved in previous session
- Code now correctly uses `chunks` and `entities` tables

### 2. No Content Being Fetched
**Issue:** Documents have no content, making processing impossible
- Notion API token is invalid
- Documents created with `content: null`
- **Impact:** Can't chunk or extract entities without content
- **Fix Required:** Valid API tokens and content fetching logic

### 3. Not Using CocoIndex
**Issue:** Manual orchestration instead of CocoIndex flows
- Processing pipeline uses imperative code
- No incremental processing
- No automatic retries or error recovery
- **Impact:** Fragile, inefficient, not production-ready
- **Fix Required:** Implement hybrid CocoIndex approach

## 🟡 FUNCTIONAL GAPS - Features Not Working

### 1. Manual Review & Editing Workflow
**Status:** UI exists but backend endpoints incomplete
- Chunk CRUD operations not fully implemented
- Entity CRUD operations not fully implemented
- Rechunking functionality not connected

### 2. Vector & Graph Storage
**Status:** Services initialized but not integrated
- Embeddings not being generated for chunks
- Vectors not being stored in Qdrant
- Entities/relationships not being stored in Neo4j

### 3. Security & Access Control
**Status:** Model exists but not enforced
- 5-tier security levels defined but not applied
- Document access control not implemented
- API authentication missing

## 🎯 REVISED PRIORITIES - Contextual Retrieval with CocoIndex

### Phase 1: Contextual Chunking Implementation (TODAY - 4-6 hours)
**Goal:** Implement Anthropic's Contextual Retrieval approach for superior search accuracy

Based on [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) research:
- Standard RAG reduces retrieval failure by 35%
- With BM25 hybrid search: 49% reduction
- With reranking: 67% reduction in retrieval failures

```python
# app/flows/document_processor.py
import cocoindex

@cocoindex.flow_def(name="DocumentProcessor")
def process_document_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    # Load document from Supabase
    data_scope["doc"] = flow_builder.add_source(
        SingleDocumentSource(document_id)
    )
    
    # Add collectors
    chunks_collector = data_scope.add_collector()
    entities_collector = data_scope.add_collector()
    
    # Transform document
    with data_scope["doc"].row() as doc:
        # Parse content if needed
        doc["content"] = doc["raw_content"].transform(
            LlamaParseFunction() if needs_parsing else PassThrough()
        )
        
        # Create hierarchical chunks
        # Level 1: Page-level chunks (1200 tokens)
        doc["page_chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=1200,
            chunk_overlap=200
        )
        
        with doc["page_chunks"].row() as page_chunk:
            # Level 2: Paragraph chunks (300-500 tokens)
            page_chunk["para_chunks"] = page_chunk["text"].transform(
                cocoindex.functions.SplitRecursively(),
                language="markdown",
                chunk_size=400,
                chunk_overlap=100
            )
            
            with page_chunk["para_chunks"].row() as para_chunk:
                # Generate contextual enrichment (50-100 tokens)
                para_chunk["context"] = para_chunk["text"].transform(
                    cocoindex.functions.ExtractByLlm(
                        llm_spec=cocoindex.LlmSpec(
                            api_type=cocoindex.LlmApiType.ANTHROPIC,
                            model="claude-3-haiku"
                        ),
                        instruction="""<document>
{doc_title}
{page_context}
</document>

<chunk>
{chunk_text}
</chunk>

Please provide a brief contextual description (50-100 tokens) that explains what this chunk is about 
and how it relates to the document. Include key entities, time periods, and relationships."""
                    )
                )
                
                # Combine context with original chunk
                para_chunk["contextualized_text"] = combine_context_and_chunk(
                    para_chunk["context"], 
                    para_chunk["text"]
                )
                
                # Generate embeddings for contextualized chunk
                para_chunk["embedding"] = para_chunk["contextualized_text"].transform(
                    cocoindex.functions.EmbedText(
                        model="text-embedding-3-small"
                    )
                )
                
                # Also store BM25 index for hybrid search
                para_chunk["bm25_tokens"] = para_chunk["contextualized_text"].transform(
                    TokenizeForBM25()
                )
                
                chunks_collector.collect(
                    document_id=doc["id"],
                    chunk_text=para_chunk["text"],
                    contextual_summary=para_chunk["context"],
                    contextualized_text=para_chunk["contextualized_text"],
                    chunk_level="paragraph",
                    parent_chunk_id=page_chunk["id"],
                    chunk_index=para_chunk["index"],
                    embedding=para_chunk["embedding"],
                    bm25_tokens=para_chunk["bm25_tokens"]
                )
        
        # Extract entities
        doc["entities"] = doc["content"].transform(
            cocoindex.functions.ExtractByLlm(
                llm_spec=cocoindex.LlmSpec(
                    api_type=cocoindex.LlmApiType.OPENAI,
                    model="gpt-4"
                ),
                output_type=list[Entity]
            )
        )
        
        with doc["entities"].row() as entity:
            entities_collector.collect(
                document_id=doc["id"],
                entity_name=entity["name"],
                entity_type=entity["type"],
                confidence=entity["confidence"]
            )
    
    # Export to Supabase (keep existing structure)
    chunks_collector.export("chunks",
        cocoindex.targets.Postgres(
            connection=supabase_conn,
            table_name="chunks"
        ))
    
    entities_collector.export("entities",
        cocoindex.targets.Postgres(
            connection=supabase_conn,
            table_name="entities"
        ))

# Keep existing API endpoint
@router.post("/documents/{id}/process")
async def process_document(id: str, background_tasks: BackgroundTasks):
    # Call CocoIndex flow instead of manual processing
    background_tasks.add_task(run_cocoindex_flow, id)
    return {"status": "processing"}

async def run_cocoindex_flow(document_id: str):
    flow = create_document_flow(document_id)
    flow.run()  # CocoIndex handles everything
```

**Tasks:**
1. Install CocoIndex properly: `pip install cocoindex`
2. Create `app/flows/document_processor.py` with CocoIndex flow
3. Create wrapper for single document source
4. Update `process_document_pipeline` to call CocoIndex
5. Test with NC2050 document

### Phase 2: Add Vector & Graph Storage (TOMORROW - 4-6 hours)
**Goal:** Store embeddings in Qdrant and entities in Neo4j

```python
# Add to the flow
chunks_collector.export("vectors",
    cocoindex.targets.Qdrant(
        connection=qdrant_conn,
        collection_name="documents",
        vector_field="embedding"
    ))

entities_collector.export("graph",
    cocoindex.targets.Neo4j(
        connection=neo4j_conn,
        mapping=cocoindex.targets.Nodes(
            label="Entity",
            key_property="id"
        )
    ))
```

### Phase 3: Fix Content Fetching (1-2 hours)
1. Update Notion API token in `.env`
2. Add content fetching fallback in processing
3. Re-scan existing documents to populate content

### Phase 4: Complete CRUD Operations (2-4 hours)
- Keep existing endpoints but ensure they work with CocoIndex-processed data
- Frontend remains unchanged

## 📊 SUCCESS METRICS

### Today's Goal
- [x] All services running
- [ ] NC2050 document has content
- [ ] NC2050 has chunks in database
- [ ] NC2050 has entities in database
- [ ] Data visible in frontend

### This Week's Goal
- [ ] All 4 documents fully processed
- [ ] Manual review workflow functional
- [ ] Vector search working
- [ ] Knowledge graph populated

## 🚀 IMPLEMENTATION STEPS

### Step 1: Create CocoIndex Wrapper (1 hour)
```python
# app/flows/sources.py
import cocoindex
from app.services.supabase_service import SupabaseService

class SingleDocumentSource(cocoindex.Source):
    def __init__(self, document_id: str):
        self.document_id = document_id
        self.supabase = SupabaseService()
    
    def fetch(self):
        doc = self.supabase.get_document(self.document_id)
        # Ensure content exists
        if not doc.content:
            # Fetch from original source
            doc.content = self.fetch_content_from_source(doc)
        return [doc.dict()]
    
    def fetch_content_from_source(self, doc):
        if doc.source_type == "notion":
            # Use NotionConnector to fetch
            pass
        elif doc.source_type == "google_drive":
            # Use GoogleDriveConnector to fetch
            pass
        return "Placeholder content for testing"
```

### Step 2: Create CocoIndex Flow (2 hours)
- Implement the flow definition above
- Add proper error handling
- Add logging for debugging

### Step 3: Update API Endpoint (30 min)
- Modify `process_document_pipeline` to call CocoIndex
- Keep same API contract for frontend

### Step 4: Test with NC2050 (1 hour)
- Trigger processing for NC2050
- Verify chunks created
- Verify entities extracted
- Check frontend display

## 📝 TESTING CHECKLIST

### Manual Testing Script (Updated for Hybrid Approach)
1. Start all services (Redis, Qdrant, Neo4j, Backend, Frontend) ✅
2. Click "Process" on NC2050 document
3. Check logs for CocoIndex flow execution
4. Wait for status to change to "pending_review"
5. Click on document to view details
6. **VERIFY**: Chunks are visible
7. **VERIFY**: Entities are visible
8. **VERIFY**: Content is displayed
9. Edit a chunk and save
10. Refresh page and verify edit persisted

### Expected Behavior with CocoIndex
- Processing should show incremental progress
- Only changed data should be reprocessed on second run
- Errors should automatically retry with backoff
- All databases should stay in sync

## 💡 KEY INSIGHTS

### Why Hybrid Approach?
1. **No Frontend Changes**: Keep existing API structure
2. **Immediate Benefits**: Get incremental processing today
3. **Low Risk**: Can fallback to manual if needed
4. **Gradual Migration**: Move other parts to CocoIndex over time

### What Changes with CocoIndex?
- **Before**: 1000+ lines of orchestration code
- **After**: ~200 lines of flow definition
- **Before**: Process all documents every time
- **After**: Only process changed documents
- **Before**: Manual error handling
- **After**: Automatic retries with backoff

## 📅 TIMELINE

### Today (Dec 24)
- [ ] Morning: Create CocoIndex wrapper and flow
- [ ] Afternoon: Test with NC2050
- [ ] Evening: Document has chunks and entities

### Tomorrow (Dec 25)
- [ ] Add Qdrant vector storage
- [ ] Add Neo4j graph storage
- [ ] Test search functionality

### This Week
- [ ] Complete manual review workflow
- [ ] Process all 4 documents
- [ ] Deploy to production

---

**Remember:** The goal is to get ONE document (NC2050) through the ENTIRE pipeline today using CocoIndex!