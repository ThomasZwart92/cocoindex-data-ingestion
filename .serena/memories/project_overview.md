# CocoIndex Data Ingestion Portal

## Project Purpose
CocoIndex is an ultra-performant data transformation framework for AI, with core engine written in Rust. This specific implementation is a data ingestion portal that:
- Processes documents from multiple sources (Notion, Google Drive, local files)
- Performs three-tier contextual chunking (Page → Paragraph → Semantic)
- Extracts entities and relationships for knowledge graphs
- Generates embeddings for semantic search
- Provides human-in-the-loop quality control with review workflow

## Tech Stack
### Backend
- **Language**: Python 3.x
- **Framework**: FastAPI for REST APIs
- **Async Processing**: Celery with Redis queue
- **Core Engine**: CocoIndex (Rust-based data transformation)

### Frontend
- **Framework**: Next.js (React)
- **Styling**: Tailwind CSS
- **Location**: `/frontend` directory

### Databases
- **PostgreSQL**: Primary database via Supabase (document states, user data)
- **Redis**: Task queue for Celery
- **Qdrant**: Vector database for embeddings
- **Neo4j**: Graph database for knowledge relationships

### Document Processing
- **LlamaParse**: PDF and document parsing
- **OpenAI/Anthropic/Google**: LLM integration for entity extraction
- **Sentence Transformers**: Embedding generation

## Architecture Pattern
The project follows CocoIndex's declarative dataflow pattern:
- All transformations are declarative, not imperative
- Each transformation creates new fields without mutation
- Full data lineage tracking
- Incremental processing support

## Current Status
✅ Completed:
- Three-tier chunking implementation
- Frontend running on localhost:3000
- Document state management
- Source connectors (Notion, Google Drive)

⚠️ In Progress:
- Embedding generation for search
- Hybrid search implementation (semantic + BM25)
- Entity extraction and storage
- Review workflow completion