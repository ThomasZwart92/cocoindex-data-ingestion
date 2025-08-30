# CocoIndex Data Ingestion Portal - Development Context

## CRITICAL RULES - NEVER VIOLATE THESE
1. **NEVER USE MOCK DATA** - Always use real API integrations
2. **ALWAYS USE REAL CONNECTORS** - NotionConnector and GoogleDriveConnector must connect to actual services
3. **NO FAKE DOCUMENTS** - Do not create dummy/sample documents
4. **REAL PROCESSING ONLY** - All background tasks must perform actual work
5. **USE EXISTING API KEYS** - Notion and Google Drive credentials are in .env file
6. **ALWAYS USE COCOINDEX DATAFLOW PATTERN** - Never use imperative service calls for processing
7. **TRACK DOCUMENT STATE** - Every document must go through the state machine lifecycle
8. **NO PLACEHOLDERS IN PRODUCTION** - Replace all placeholder implementations with real code
9. **TABLE NAMING CONSISTENCY** - Supabase has 'chunks' and 'entities' tables (NOT 'document_chunks' or 'document_entities')

## Project Overview
Building a high-quality document ingestion system for < 100 documents with human-in-the-loop review, leveraging CocoIndex for processing.

## 🚨 CURRENT STATUS (2025-08-25)
**PROGRESS**: Core pipeline ~75% complete, three-tier chunking implemented
- ✅ Three-tier hierarchical chunking with contextual enrichment working
- ✅ Documents process and chunks persist correctly to database
- ✅ LlamaParse integration implemented and working
- ✅ Source connectors fetch documents successfully
- ✅ Frontend shows processing status with hierarchical chunk display
- ✅ Entity extraction improved - filters out common words, focuses on meaningful entities
- ✅ Docker deployment configuration ready

**COMPLETED IMPROVEMENTS**:
1. **Three-Tier Chunking**: Page → Paragraph → Semantic levels implemented
2. **Contextual Summaries**: AI-generated context for each chunk
3. **Semantic Focus**: Each semantic chunk tagged with its key concept
4. **Database Schema**: Fixed table naming and field mappings
5. **Entity Extraction**: Enhanced prompts and filtering to extract only meaningful entities
6. **Deployment Ready**: Docker and docker-compose configurations for full stack

**Next Priority**: Generate embeddings for contextualized chunks and implement hybrid search

## Key Project Documents
- **CocoIndex README**: @README.md

## Critical Reference Files

### Core CocoIndex Components
- `python/cocoindex/flow.py` - Flow definition and builder classes
- `python/cocoindex/sources.py` - Source connectors (GoogleDrive, LocalFile, etc.)
- `python/cocoindex/targets.py` - Target databases (Postgres, Qdrant, Neo4j)
- `python/cocoindex/functions.py` - Built-in functions (chunking, embeddings)
- `python/cocoindex/llm.py` - LLM integration specs

### Essential Examples
- `examples/docs_to_knowledge_graph/main.py` - Entity extraction and Neo4j integration
- `examples/multi_format_indexing/main.py` - PDF/image processing with ColPali
- `examples/gdrive_text_embedding/main.py` - Google Drive integration pattern
- `examples/text_embedding_qdrant/main.py` - Qdrant vector storage pattern

### Key Functions Documentation
- `docs/docs/ops/functions.md` - Complete function reference

## CocoIndex Key Concepts & Best Practices

### 1. Dataflow Programming Model
- **No Mutations**: Each transformation creates a new field, never modify existing ones
- **Observable Data**: All data before/after transformations is visible with lineage
- **Declarative**: Define transformations, not imperative operations

```python
# GOOD: Create new fields
doc["chunks"] = doc["content"].transform(...)
chunk["embedding"] = chunk["text"].transform(...)

# BAD: Never mutate in place
doc["content"] = process(doc["content"])  # Don't do this
```

### 2. Flow Definition Pattern
```python
@cocoindex.flow_def(name="FlowName")
def flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    # 1. Add sources
    data_scope["documents"] = flow_builder.add_source(...)
    
    # 2. Add collectors
    output = data_scope.add_collector()
    
    # 3. Transform with row context
    with data_scope["documents"].row() as doc:
        doc["processed"] = doc["content"].transform(...)
        output.collect(...)
    
    # 4. Export to targets
    output.export("name", target_spec)
```

### 3. Chunking Strategies - Three-Tier Contextual Retrieval

Based on [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) research, we implement an enhanced three-tier hierarchical chunking strategy with contextual enrichment for maximum retrieval accuracy.

#### Three-Tier Hierarchical Structure (IMPLEMENTED)
```python
# Level 1: Page-level chunks (1200 tokens) - Document overview
doc["page_chunks"] = doc["content"].transform(
    cocoindex.functions.SplitRecursively(),
    language="markdown",
    chunk_size=1200,
    chunk_overlap=200
)

# Level 2: Paragraph chunks (200-400 tokens) - Balanced retrieval
with doc["page_chunks"].row() as page:
    page["para_chunks"] = page["text"].transform(
        cocoindex.functions.SplitRecursively(),
        chunk_size=300,
        chunk_overlap=50
    )
    
    # Level 3: Semantic chunks (20-100 tokens) - Ultra-precise vectors
    with page["para_chunks"].row() as paragraph:
        paragraph["semantic_chunks"] = paragraph["text"].transform(
            SplitBySentences(),  # 1-3 sentences per chunk
            max_tokens=100,
            identify_focus=True  # Tags each chunk with semantic focus
        )
        
        # Add contextual enrichment to all levels
        with paragraph["semantic_chunks"].row() as chunk:
            chunk["context"] = chunk["text"].transform(
                GenerateChunkContext(doc_title, paragraph_context)
            )
            chunk["contextualized"] = f"{chunk['context']}\n\n{chunk['text']}"
```

#### Why Three Tiers?
- **Page chunks**: Broad context for document-level queries
- **Paragraph chunks**: Balanced for standard retrieval
- **Semantic chunks**: Single-concept precision (solves vector pollution problem)

#### Hybrid Search Strategy
1. **Semantic Search**: Embeddings of contextualized chunks
2. **BM25 Search**: Lexical matching on contextualized text
3. **Fusion**: Combine results from both methods
4. **Reranking**: Use Cohere or similar to rerank top 150 → top 20

#### Performance Improvements
- Contextual chunks alone: 35% reduction in retrieval failures
- + BM25 hybrid: 49% reduction
- + Reranking: 67% reduction

### 4. LLM Integration
```python
# Entity extraction
doc["entities"] = doc["content"].transform(
    cocoindex.functions.ExtractByLlm(
        llm_spec=cocoindex.LlmSpec(
            api_type=cocoindex.LlmApiType.OPENAI,
            model="gpt-4o"
        ),
        output_type=list[Entity],  # Your dataclass
        instruction="Extract entities..."
    )
)
```

**Important LLM Provider Notes:**
- **Gemini Token Requirements**: Minimum 1000 tokens for any output (auto-adjusted in our implementation)
- **JSON Parsing**: Handle both raw JSON (OpenAI) and markdown-wrapped JSON (Gemini)
- **Model Updates**: Using gemini-2.5-pro (not 1.5-flash which is outdated)
- **Fallback Support**: Automatic fallback between OpenAI and Gemini on failures

### 5. Multi-Database Architecture
```python
# Vector storage (Qdrant)
output.export("vectors", 
    cocoindex.targets.Qdrant(
        connection=qdrant_conn,
        collection_name="docs"
    ))

# Knowledge Graph (Neo4j)
output.export("graph",
    cocoindex.targets.Neo4j(
        connection=neo4j_conn,
        mapping=cocoindex.targets.Nodes(label="Document")
    ))

# Metadata (Postgres/Supabase)
output.export("metadata",
    cocoindex.targets.Postgres(
        connection=postgres_conn
    ))
```

### 6. Source Connectors

#### Google Drive
```python
flow_builder.add_source(
    cocoindex.sources.GoogleDrive(
        service_account_credential_path="path/to/creds.json",
        root_folder_ids=["folder_id"],
        recent_changes_poll_interval=timedelta(minutes=30)
    )
)
```

#### Notion (Not Native - Custom Implementation Needed)
- Use Notion API directly
- Convert to CocoIndex-compatible format
- Feed into flow as custom source

### 7. Document Parsing & Image Handling

#### LlamaParse Three-Tier Strategy:
```python
from llama_parse import LlamaParse

class ParseTier(Enum):
    BALANCED = "balanced"        # $0.10/page - Start here
    AGENTIC = "agentic"          # $0.40/page - Complex layouts
    AGENTIC_PLUS = "agentic_plus"  # $1.00/page - Critical docs

@cocoindex.op.function()
def parse_document(content: bytes, tier: str = "balanced") -> tuple[str, dict]:
    parser = LlamaParse(
        api_key="...",
        result_type="markdown",
        parsing_mode=tier
    )
    
    result = parser.parse_raw(content)
    # LlamaParse extracts images and includes descriptions in markdown
    return result.text, {"tier": tier, "needs_review": True}
```

#### Image Processing Pipeline:
```python
@cocoindex.op.function()
def extract_and_process_images(content: bytes, doc_text: str) -> list[dict]:
    """Extract images and generate rich metadata"""
    images = []
    
    # Extract images from PDF
    from pdf2image import convert_from_bytes
    pdf_images = convert_from_bytes(content, dpi=300)
    
    for idx, img in enumerate(pdf_images):
        # Convert to bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format="PNG")
        
        # Generate caption with Vision API
        caption = generate_image_caption(img_bytes.getvalue(), doc_text[:1000])
        
        # Generate embeddings
        visual_embedding = ColPaliEmbedImage(model="vidore/colpali-v1.2")
        text_embedding = EmbedText(model="text-embedding-3-small")
        
        images.append({
            "page": idx + 1,
            "bytes": img_bytes.getvalue(),
            "caption": caption,
            "visual_embedding": visual_embedding(img_bytes.getvalue()),
            "text_embedding": text_embedding(caption),
            "needs_review": True
        })
    
    return images

def generate_image_caption(image_bytes: bytes, context: str) -> str:
    """Use GPT-4 Vision to generate detailed captions"""
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context: {context}\nDescribe this image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"}}
            ]
        }]
    )
    return response.choices[0].message.content
```

#### Storage Strategy:
```python
# Store images with multiple vectors in Qdrant
image_output.export("document_images",
    cocoindex.targets.Qdrant(
        connection=qdrant_conn,
        collection_name="images",
        vectors=[
            {"name": "visual", "field": "visual_embedding", "size": 1024},
            {"name": "text", "field": "text_embedding", "size": 1536}
        ]
    ),
    metadata={
        "document_id": doc["id"],
        "page_number": img["page"],
        "caption": img["caption"],
        "storage_url": img["supabase_url"]
    }
)
```

#### Search Capabilities:
- **Text search**: Find images by caption content
- **Visual search**: Find similar images (ColPali)
- **Hybrid search**: Combine both approaches
- **Context-aware**: Images linked to document chunks

### 8. Error Handling & Retries
- CocoIndex handles incremental processing automatically
- Failed items can be retried
- State is persisted in Postgres

### 9. Authentication Pattern
```python
# Store credentials securely
conn = cocoindex.add_auth_entry(
    "connection_name",
    cocoindex.targets.QdrantConnection(
        grpc_url="http://localhost:6334"
    )
)
```

## Implementation Critical Path

### Phase 0: Foundation (BLOCKING)
**Without these, nothing works:**
1. Async infrastructure (Celery + Redis)
2. State machine for document lifecycle
3. Idempotent operations
4. Error handling with retries

### Must-Have Quality Features
- Multiple chunking strategies
- Manual chunk editing
- Entity correction
- Custom metadata fields
- Multi-model comparison (GPT-4 vs Gemini)

### Can Defer (Scale Features)
- Batch operations
- Auto-approval
- WebSockets
- Complex caching

## Common Pitfalls to Avoid

1. **Don't Skip Async**: Without Celery/Redis, app will hang during 30+ second operations
2. **Don't Skip State Management**: Will lose track of documents
3. **Don't Build UI First**: Pipeline must work before adding interface
4. **Don't Use pgvector**: Qdrant has 2-3x better latency for chatbot use case
5. **Don't Skip Neo4j**: Knowledge graphs are worth the complexity even at small scale
6. **Don't Ignore Images**: They contain critical information - captions and OCR are essential
7. **Don't Auto-approve Captions**: AI-generated image descriptions need human review for quality
8. **Gemini Token Minimum**: Gemini requires at least 1000 tokens to generate any output (unlike OpenAI which works with 50+)
9. **JSON Response Parsing**: Gemini often wraps JSON in markdown code blocks (```json...```), while OpenAI returns raw JSON
10. **@flow_def Decorator**: Functions decorated with `@cocoindex.flow_def` become Flow objects, not callable functions
11. **LocalMemory Doesn't Exist**: CocoIndex only has file-based sources (LocalFile, GoogleDrive, AmazonS3, AzureBlob)
12. **Not Everything Needs a Flow**: Single-document, synchronous operations don't benefit from CocoIndex flows

## When to Use CocoIndex Flows vs Direct Implementation

### Use CocoIndex Flows When:
- **Batch Processing**: Processing multiple documents/files
- **Incremental Updates**: Need to track changes and only process new/modified items
- **Data Lineage**: Need visibility into transformation pipeline
- **Multiple Targets**: Exporting to multiple databases (Qdrant + Neo4j + Postgres)
- **Complex Pipelines**: Multi-stage transformations with nested operations
- **Continuous Monitoring**: Watching folders for changes with refresh intervals

### Use Direct Implementation When:
- **Single Document Operations**: Processing one item at a time (like metadata extraction)
- **Synchronous Operations**: Need immediate results without async processing
- **Simple Transformations**: Single-step operations without complex pipelines
- **In-Memory Data**: Working with data already in memory (not from files)
- **API Endpoints**: Direct request-response patterns
- **No State Tracking Needed**: One-time operations without incremental updates

### CocoIndex Flow Instantiation Patterns

#### Pattern 1: Using @flow_def decorator (Recommended)
```python
@cocoindex.flow_def(name="MyFlow")
def my_flow(flow_builder: FlowBuilder, data_scope: DataScope):
    # Define flow logic
    pass

# The decorated function IS the flow object
flow = my_flow  # NOT my_flow() - it's already a Flow object!
flow.reset()
flow.run()
```

#### Pattern 2: Manual flow building (for dynamic flows)
```python
def build_custom_flow():
    flow_builder = cocoindex.FlowBuilder("CustomFlow")
    data_scope = cocoindex.DataScope()
    
    # Add sources and transformations
    data_scope["data"] = flow_builder.add_source(...)
    
    return flow_builder.build()

flow = build_custom_flow()
flow.run()
```

### Available CocoIndex Sources
- `cocoindex.sources.LocalFile(path="...")` - Read files from local filesystem
- `cocoindex.sources.GoogleDrive(...)` - Read from Google Drive
- `cocoindex.sources.AmazonS3(...)` - Read from S3 buckets
- `cocoindex.sources.AzureBlob(...)` - Read from Azure Blob Storage

**Note**: There is NO `LocalMemory` source. For in-memory data, use direct implementation.

## Development Commands

```bash
# Initialize CocoIndex
import cocoindex
cocoindex.init()

# Run a flow (when using @flow_def decorator)
flow = text_embedding_flow  # Note: NO parentheses - it's already a Flow object
flow.reset()  # Clear any previous state
flow.run()

# Check incremental processing
flow.run()  # Only processes changes

# For debugging flow issues
print(type(my_flow))  # Should show <class 'cocoindex.Flow'> if decorated
```

## Key Architecture Decisions

### Why These Technologies:
- **CocoIndex**: Handles incremental processing, lineage tracking
- **LlamaParse**: AI-powered document parsing with quality tiers (perfect for low-volume, high-quality)
- **Qdrant > pgvector**: Better latency, native multi-vector support
- **Neo4j**: Powerful relationship queries for context
- **Celery + Redis**: Essential for async processing
- **Supabase**: Auth + PostgreSQL + storage in one

### Document Processing State Machine:
```
DISCOVERED → PROCESSING → PENDING_REVIEW → APPROVED → INGESTED
                ↓             ↓             ↓
             FAILED       REJECTED      FAILED
```

## Quality Metrics for Success
- 100% chunks reviewed before storage
- 100% entities validated
- 0% data loss on failures
- < 30s processing per document
- < 200ms vector search latency

## Docker Deployment

### Quick Start
```bash
# Start all services
docker-compose -f docker-compose.full.yml up -d

# Check service health
docker-compose -f docker-compose.full.yml ps

# View logs
docker-compose -f docker-compose.full.yml logs -f backend

# Stop all services
docker-compose -f docker-compose.full.yml down
```

### Service Architecture
- **Frontend**: Nginx serving React app on port 3000
- **Backend**: FastAPI + Uvicorn on port 8001
- **Celery Worker**: Background task processing
- **Postgres**: Primary database on port 5432
- **Redis**: Cache and Celery broker on port 6379
- **Neo4j**: Knowledge graph on ports 7474/7687
- **Qdrant**: Vector database on port 6333
- **Supabase**: External service (not dockerized)

### Environment Variables
Required in `.env` file:
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `NOTION_API_KEY`
- `GOOGLE_DRIVE_CREDENTIALS_PATH`
- `LLAMAPARSE_API_KEY`
- `NEO4J_PASSWORD`
- `QDRANT_API_KEY`

### Deployment Options
1. **Local Development**: Use docker-compose.yml (services only)
2. **Full Stack Local**: Use docker-compose.full.yml (includes app)
3. **Production**: Deploy to cloud with managed services
   - Use managed Postgres (RDS, Cloud SQL)
   - Use managed Redis (ElastiCache, Cloud Memorystore)
   - Deploy backend on Cloud Run/App Engine/ECS
   - Deploy frontend on CDN (CloudFront, Cloudflare)

## Remember
**Quality > Features > Scale**

Build simple and working first, add complexity only when needed.