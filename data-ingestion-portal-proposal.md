# Minimalist Data Ingestion Review Portal - Design Proposal

## Executive Summary

A minimalist, function-first web portal for managing **low-volume, high-quality** data ingestion from Notion and Google Drive sources, with comprehensive review and approval workflows before processing through CocoIndex into vector and graph databases.

### Optimized for Low Document Volume
This design is specifically tailored for organizations processing **< 100 documents** with emphasis on:
- **Quality over quantity**: Extensive metadata and entity extraction
- **Human-in-the-loop**: Every chunk can be reviewed and refined
- **Simplified architecture**: No unnecessary complexity for small-scale needs

## Visual Design System

### Design Principles
- **Brutalist minimalism**: Function over form, raw data visibility
- **Monospace typography**: `JetBrains Mono` or `IBM Plex Mono`
- **High contrast**: Pure black (#000) on white (#FFF), single accent color (#0066FF)
- **Dense information**: Maximize data per screen
- **Keyboard-first**: All actions accessible via shortcuts

## Application Architecture

### Simplified Tech Stack for Low Volume
```yaml
# CRITICAL - Must Have (Week 1-2)
Frontend:
  - Next.js 14 (App Router)
  - Tailwind CSS (minimal custom styles)
  - React Query (data fetching)

Backend:
  - FastAPI
  - CocoIndex (processing engine)
  - Celery + Redis (CRITICAL: Async processing)
  - SQLAlchemy (CRITICAL: State management)

Databases:
  - Supabase (PostgreSQL + Auth + State storage)
  - Qdrant (Vector storage - better latency than pgvector)
  - Neo4j AuraDB (Knowledge graph)

Infrastructure:
  - Railway (Hosting)
  - Redis Cloud (Job queue)

# DEFER - Scale Features Only (Post-MVP)
- WebSockets (use polling for now)
- Radix UI (use native HTML)
- Zustand (use React Query cache)
- Google Cloud Storage (use Supabase storage)
- Batch operations (quality doesn't need bulk processing)
- Auto-approval (human review ensures quality)
```

### Critical Architectural Components

#### 1. Async Processing (MUST HAVE)
```python
# Without this, your app will hang during processing
from celery import Celery

celery_app = Celery('ingestion', broker='redis://...')

@celery_app.task
def process_document(doc_id: str):
    """Process document asynchronously"""
    # Long-running CocoIndex operations here
    pass
```

#### 2. State Management (MUST HAVE)
```python
# Track document processing states
class DocumentState(Enum):
    DISCOVERED = "discovered"
    PROCESSING = "processing" 
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    INGESTED = "ingested"
    FAILED = "failed"
```

#### 3. Idempotent Operations (MUST HAVE)
```python
# Prevent duplicate processing on retries
@celery_app.task(bind=True, max_retries=3)
def process_chunk(self, chunk_id: str):
    if chunk_exists(chunk_id):
        return get_existing_chunk(chunk_id)
    # Process new chunk...
```

## Page Layouts

### 1. Dashboard (`/`)

```
┌─────────────────────────────────────────────────────────────┐
│ INGESTION CONTROL                            [user@email]    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ SYSTEM STATUS                                                │
│ ┌──────────────┬──────────────┬──────────────┬────────────┐│
│ │ PostgreSQL   │ Qdrant       │ Neo4j        │ CocoIndex  ││
│ │ ● ONLINE     │ ● ONLINE     │ ● ONLINE     │ ● IDLE     ││
│ │ 12ms         │ 8ms          │ 15ms         │ 0 jobs     ││
│ └──────────────┴──────────────┴──────────────┴────────────┘│
│                                                              │
│ SOURCES                                                      │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ Notion:  12 pages connected   [LAST SCAN: 2 min ago]     ││
│ │ GDrive:  8 docs connected     [LAST SCAN: 5 min ago]     ││
│ │                                                           ││
│ │ [3] new documents pending approval →                     ││
│ └──────────────────────────────────────────────────────────┘│
│                                                              │
│ DOCUMENTS                                          [+ ADD]  │
│ ┌──────────────────────────────────────────────────────────┐│
│ │ ID    SOURCE  TITLE                    CHUNKS  STATUS    ││
│ │ ────────────────────────────────────────────────────────││
│ │ d001  notion  Product Strategy 2024    42      active    ││
│ │ d002  gdrive  Q4 Financial Report      28      active    ││
│ │ d003  notion  Engineering Handbook     156     active    ││
│ │ d004  gdrive  Customer Feedback        12      pending   ││
│ │                                                          ││
│ │ [TAB] select  [E] edit  [D] delete  [R] rechunk         ││
│ └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2. Document Detail/Edit (`/document/{id}`)

```
┌─────────────────────────────────────────────────────────────┐
│ ← BACK     DOCUMENT: Product Strategy 2024          [SAVE]  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌─[METADATA]────────────────────────────────────────────┐  │
│ │ source:      notion                                    │  │
│ │ source_id:   page_12345                               │  │
│ │ author:      [John Smith               ] ← AI suggested│  │
│ │ category:    [strategy                 ]              │  │
│ │ tags:        [product, 2024, roadmap   ]              │  │
│ │ department:  [product                  ]              │  │
│ │ visibility:  [internal                 ]              │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[CHUNKING]────────────────────────────────────────────┐  │
│ │ method:      [recursive         ▼]                    │  │
│ │ size:        [1500] bytes     overlap: [200]         │  │
│ │ min_size:    [500 ] bytes                            │  │
│ │ language:    [markdown     ▼]                        │  │
│ │                                    [RECHUNK]         │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[CHUNKS]──────────────────────────────────────────────┐  │
│ │ #  SIZE   CONTENT                           [ACTIONS] │  │
│ │ ─────────────────────────────────────────────────────│  │
│ │ 1  1487   # Product Strategy 2024...        [E] [X]  │  │
│ │ 2  1502   ## Market Analysis...             [E] [X]  │  │
│ │ 3  1456   Our primary focus...              [E] [X]  │  │
│ │            ┌─[CONTEXT]──────────────┐                 │  │
│ │            │ BEFORE: ...strategy    │                 │  │
│ │            │ MAIN: Our primary...   │                 │  │
│ │            │ AFTER: The next phase..│                 │  │
│ │            └────────────────────────┘                 │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[ENTITIES]────────────────────────────────────────────┐  │
│ │ TYPE      NAME              CONFIDENCE  [APPROVE ALL] │  │
│ │ ─────────────────────────────────────────────────────│  │
│ │ Person    John Smith        0.95        [✓] [ ]      │  │
│ │ Company   Acme Corp         0.88        [✓] [ ]      │  │
│ │ Project   Atlas             0.92        [✓] [ ]      │  │
│ │ Concept   Machine Learning  0.97        [✓] [ ]      │  │
│ └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3. Approval Queue (`/queue`)

```
┌─────────────────────────────────────────────────────────────┐
│ APPROVAL QUEUE                              [3] pending     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌─[NEW DOCUMENTS]───────────────────────────────────────┐  │
│ │ SOURCE  TITLE                    DETECTED   [ACTION]   │  │
│ │ ───────────────────────────────────────────────────   │  │
│ │ notion  Marketing Plan Q1        2 min ago  [REVIEW]  │  │
│ │ gdrive  Budget Forecast.xlsx     15 min ago [REVIEW]  │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[UPDATED DOCUMENTS]───────────────────────────────────┐  │
│ │ SOURCE  TITLE            CHANGES         [ACTION]      │  │
│ │ ──────────────────────────────────────────────────    │  │
│ │ notion  Product Roadmap  +12 lines       [REVIEW]     │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[DOCUMENT PREVIEW: Marketing Plan Q1]─────────────────┐  │
│ │                                                        │  │
│ │ PROPOSED CHUNKS: 18                                    │  │
│ │ ┌────────────────────────────────────────────────┐   │  │
│ │ │ 1. Executive Summary                           │   │  │
│ │ │ 2. Target Audience Analysis                    │   │  │
│ │ │ 3. Campaign Strategy                           │   │  │
│ │ └────────────────────────────────────────────────┘   │  │
│ │                                                        │  │
│ │ EXTRACTED ENTITIES: 24                                │  │
│ │ • 8 People  • 5 Companies  • 11 Concepts             │  │
│ │                                                        │  │
│ │ AI SUGGESTED METADATA:                                │  │
│ │ • Department: Marketing                               │  │
│ │ • Quarter: Q1 2024                                    │  │
│ │ • Author: Sarah Johnson                               │  │
│ │                                                        │  │
│ │ [APPROVE]  [MODIFY]  [REJECT]  [SKIP]                │  │
│ └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4. Settings (`/settings`)

```
┌─────────────────────────────────────────────────────────────┐
│ SETTINGS                                                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ ┌─[SOURCE CONFIGURATION]────────────────────────────────┐  │
│ │                                                        │  │
│ │ NOTION INTEGRATION                                    │  │
│ │ API Token:    [••••••••••••••••••]          [TEST]   │  │
│ │ Workspaces:   [3] connected                          │  │
│ │ Scan Interval: [5] minutes                           │  │
│ │                                                        │  │
│ │ GOOGLE DRIVE                                          │  │
│ │ Service Acct: [ingestion@project.iam...]    [TEST]   │  │
│ │ Folders:      [/shared/docs, /team]                  │  │
│ │ File Types:   [.pdf, .docx, .txt, .md]              │  │
│ │ Scan Interval: [10] minutes                          │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[PROCESSING DEFAULTS]─────────────────────────────────┐  │
│ │                                                        │  │
│ │ Chunk Size:    [1500] bytes                          │  │
│ │ Chunk Overlap: [200] bytes                           │  │
│ │ Min Size:      [500] bytes                           │  │
│ │                                                        │  │
│ │ LLM Model:     [gpt-4o-mini        ▼]               │  │
│ │ Embedding:     [text-embedding-3-small ▼]           │  │
│ │                                                        │  │
│ │ Auto-approve confidence: [0.90]                      │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ ┌─[DATABASE CONNECTIONS]────────────────────────────────┐  │
│ │                                                        │  │
│ │ Supabase:  ✓ Connected                               │  │
│ │ Qdrant:    ✓ Connected (92,451 vectors)             │  │
│ │ Neo4j:     ✓ Connected (1,247 nodes, 3,892 edges)   │  │
│ └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Features (Prioritized for Low Volume)

### Must Have (Week 1-3)
- **Document State Tracking**: Track processing lifecycle
- **Async Processing**: Non-blocking document processing
- **Simple CRUD**: View and delete documents only
- **Error Recovery**: Retry failed processing
- **Basic Approval**: One document at a time

### Chunking Control

#### Phase 1 (Quality-Critical Features):
- **Multiple Strategies**: Recursive AND semantic chunking (quality requires flexibility)
- **Manual Editing**: Modify individual chunks (essential for quality control)
- **Preview with Context**: See chunks with surrounding text before approval
- **Hierarchical Display**: Parent-child relationships (understand document structure)
- **Custom chunk size/overlap**: Adjust parameters per document type

#### Phase 2 (Scale/Convenience):
- **Batch rechunking**: Apply settings to multiple docs
- **Chunk templates**: Save preferred settings
- **Auto-chunking rules**: Based on document type

### Metadata Management

#### Phase 1 (Quality-Critical):
- **AI Pre-population**: LLM suggests metadata (MUST HAVE)
- **Manual Override**: Edit ALL suggested values (MUST HAVE)
- **Custom Fields**: Add domain-specific metadata (quality requires flexibility)
- **Multi-model suggestions**: Use different LLMs for comparison (GPT-4 + Gemini)

#### Phase 2 (Scale/Automation):
- **Validation Rules**: Automated quality checks
- **Metadata templates**: Reusable field sets
- **Bulk metadata editing**: Apply to multiple docs

### Entity Extraction

#### Phase 1 (Essential for Knowledge Graph Quality):
- **Preview Entities**: See extracted entities before approval
- **Confidence Scores**: Display extraction confidence
- **Manual Correction**: Add/remove/edit entities (CRITICAL for quality)
- **Relationship Mapping**: Preview graph connections (understand context)
- **Entity Resolution**: Merge duplicate entities (e.g., "OpenAI" vs "Open AI")

#### Phase 2 (Scale/Automation):
- **Batch entity approval**: Apply to multiple docs
- **Entity templates**: Common entities for your domain
- **Auto-approval rules**: Based on confidence

### Source Synchronization (Simplified for Low Volume)

#### Phase 1 - Manual Trigger:
- **Notion Integration**: 
  - Connect via API token
  - Manual scan button (no auto-polling initially)
  - Simple change detection
  
- **Google Drive Integration**:
  - Service account authentication
  - Manual scan button
  - Basic file type filtering (.pdf, .docx, .txt)

#### Phase 2 - Automation:
- Scheduled polling (every 30 min is fine for low volume)
- Change detection optimization
- More file types

### Approval Workflow

#### Phase 1 (MVP):
- **Simple Queue**: New documents only
- **Preview Panel**: See chunks and entities
- **Single Approval**: One document at a time
- **Basic Rejection**: Mark as failed

#### Phase 2 (Scale Later):
- **Update Handling**: Track document changes
- **Batch Approval**: Multiple documents
- **Auto-approval Rules**: Confidence thresholds
- **Reprocessing**: Send back with feedback

## Implementation Details

### Backend Services

```python
# Source Monitor Service
class SourceMonitor:
    def __init__(self):
        self.notion_client = NotionClient()
        self.drive_client = GoogleDriveClient()
        
    async def scan_sources(self):
        """Periodic scan for new/changed documents"""
        notion_changes = await self.scan_notion()
        drive_changes = await self.scan_gdrive()
        
        for change in notion_changes + drive_changes:
            await self.queue_for_processing(change)
    
    async def scan_notion(self):
        """Check Notion for updates"""
        pages = await self.notion_client.get_pages()
        changes = []
        
        for page in pages:
            last_scan = await db.get_last_scan(page.id)
            if page.last_edited > last_scan:
                changes.append({
                    'source': 'notion',
                    'id': page.id,
                    'title': page.title,
                    'content': await page.get_content()
                })
        
        return changes
    
    async def queue_for_processing(self, document):
        """Add document to approval queue"""
        # Pre-process with CocoIndex
        chunks = await self.generate_chunks(document)
        entities = await self.extract_entities(document)
        metadata = await self.suggest_metadata(document)
        
        # Store in queue
        await db.ingestion_queue.insert({
            'document': document,
            'chunks': chunks,
            'entities': entities,
            'metadata': metadata,
            'status': 'pending'
        })
```

### Frontend Components

```typescript
// Chunk Preview Component
export function ChunkPreview({ chunk, document, onEdit }) {
  const [showContext, setShowContext] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  
  return (
    <div className="font-mono text-sm border border-black p-2">
      <div className="flex justify-between mb-2">
        <span>Chunk #{chunk.number} ({chunk.size} bytes)</span>
        <div className="space-x-2">
          <button onClick={() => setShowContext(!showContext)}>
            {showContext ? 'Hide' : 'Show'} Context
          </button>
          <button onClick={() => setIsEditing(true)}>Edit</button>
          <button onClick={() => onDelete(chunk.id)}>Delete</button>
        </div>
      </div>
      
      {showContext && (
        <div className="opacity-50 text-xs mb-2">
          ...{document.getTextBefore(chunk.start, 200)}
        </div>
      )}
      
      {isEditing ? (
        <textarea 
          value={chunk.text}
          onChange={(e) => onEdit(chunk.id, e.target.value)}
          className="w-full h-32 p-2 border border-black"
        />
      ) : (
        <div className="whitespace-pre-wrap">{chunk.text}</div>
      )}
      
      {showContext && (
        <div className="opacity-50 text-xs mt-2">
          {document.getTextAfter(chunk.end, 200)}...
        </div>
      )}
    </div>
  );
}

// Entity Preview Component
export function EntityPreview({ entities, onApprove, onEdit }) {
  return (
    <div className="font-mono text-sm">
      <table className="w-full">
        <thead>
          <tr className="border-b-2 border-black">
            <th className="text-left p-1">Type</th>
            <th className="text-left p-1">Name</th>
            <th className="text-left p-1">Confidence</th>
            <th className="text-left p-1">Actions</th>
          </tr>
        </thead>
        <tbody>
          {entities.map(entity => (
            <tr key={entity.id} className="border-b border-gray-300">
              <td className="p-1">{entity.type}</td>
              <td className="p-1">{entity.name}</td>
              <td className="p-1">{(entity.confidence * 100).toFixed(0)}%</td>
              <td className="p-1">
                <button onClick={() => onApprove(entity.id)}>✓</button>
                <button onClick={() => onEdit(entity.id)}>✎</button>
                <button onClick={() => onDelete(entity.id)}>✗</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### Keyboard Shortcuts

```javascript
// Global keyboard handler
export function useKeyboardShortcuts() {
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Command palette
      if (e.metaKey && e.key === 'k') {
        openCommandPalette();
      }
      
      // Navigation
      if (e.key === 'g') {
        if (e.shiftKey) {
          switch(e.key) {
            case 'd': navigate('/'); break;
            case 'q': navigate('/queue'); break;
            case 's': navigate('/settings'); break;
          }
        }
      }
      
      // Actions (when item selected)
      if (selectedItem && !e.metaKey && !e.ctrlKey) {
        switch(e.key) {
          case 'e': editItem(selectedItem); break;
          case 'd': deleteItem(selectedItem); break;
          case 'r': rechunkItem(selectedItem); break;
          case 'a': approveItem(selectedItem); break;
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [selectedItem]);
}
```

## Styling Guidelines

### Minimalist CSS
```css
/* Base typography */
:root {
  --font-mono: 'JetBrains Mono', 'IBM Plex Mono', monospace;
  --color-black: #000000;
  --color-white: #FFFFFF;
  --color-accent: #0066FF;
  --color-error: #FF0000;
  --color-success: #00AA00;
}

body {
  font-family: var(--font-mono);
  font-size: 14px;
  line-height: 1.4;
  background: var(--color-white);
  color: var(--color-black);
}

/* No decorative elements */
* {
  border-radius: 0 !important;
  box-shadow: none !important;
}

/* Dense data display */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.data-table td {
  padding: 0.25rem 0.5rem;
  border-bottom: 1px solid var(--color-black);
}

/* Minimal buttons */
button {
  background: var(--color-white);
  border: 2px solid var(--color-black);
  padding: 0.5rem 1rem;
  font-family: inherit;
  font-size: inherit;
  cursor: pointer;
  transition: none;
}

button:hover {
  background: var(--color-black);
  color: var(--color-white);
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Status indicators */
.status-online { color: var(--color-success); }
.status-offline { color: var(--color-error); }
.status-pending { color: var(--color-accent); }

/* No animations unless functional */
@media (prefers-reduced-motion: no-preference) {
  .loading-spinner {
    animation: spin 1s linear infinite;
  }
}
```

## Database Schema

### Supabase Tables

```sql
-- Document management
CREATE TABLE documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type VARCHAR(50) NOT NULL,
  source_id VARCHAR(255) NOT NULL,
  title TEXT NOT NULL,
  content TEXT,
  metadata JSONB DEFAULT '{}',
  ai_metadata JSONB DEFAULT '{}',
  user_metadata JSONB DEFAULT '{}',
  status VARCHAR(50) DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  ingested_at TIMESTAMPTZ,
  UNIQUE(source_type, source_id)
);

-- Ingestion queue
CREATE TABLE ingestion_queue (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id),
  action VARCHAR(50) NOT NULL, -- 'create', 'update', 'delete'
  changes JSONB,
  chunks JSONB,
  entities JSONB,
  status VARCHAR(50) DEFAULT 'pending',
  reviewed_by UUID REFERENCES auth.users(id),
  reviewed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Document chunks with vectors
CREATE TABLE document_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  chunk_number INTEGER NOT NULL,
  chunk_text TEXT NOT NULL,
  chunk_size INTEGER,
  start_position INTEGER,
  end_position INTEGER,
  embedding VECTOR(1536), -- OpenAI dimension
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(document_id, chunk_number)
);

-- Create vector index
CREATE INDEX ON document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Source configurations
CREATE TABLE source_configs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_type VARCHAR(50) NOT NULL,
  config JSONB NOT NULL,
  last_scan_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## API Endpoints

### FastAPI Routes

```python
from fastapi import FastAPI, WebSocket
from typing import List, Optional

app = FastAPI(title="Data Ingestion Portal")

# Document Management
@app.get("/api/documents")
async def list_documents(
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 100
) -> List[Document]:
    """List all documents with optional filtering"""
    pass

@app.post("/api/documents/{id}/rechunk")
async def rechunk_document(
    id: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: Optional[int] = None
) -> ChunkResult:
    """Rechunk document with new parameters"""
    pass

@app.put("/api/documents/{id}/metadata")
async def update_metadata(
    id: str,
    metadata: dict
) -> Document:
    """Update document metadata"""
    pass

# Approval Queue
@app.get("/api/queue/pending")
async def get_pending_approvals() -> List[QueueItem]:
    """Get all pending items in approval queue"""
    pass

@app.post("/api/queue/{id}/approve")
async def approve_item(id: str) -> ProcessingResult:
    """Approve item and trigger processing"""
    pass

@app.post("/api/queue/{id}/reject")
async def reject_item(
    id: str,
    reason: Optional[str] = None
) -> QueueItem:
    """Reject item with optional reason"""
    pass

# Source Management
@app.post("/api/sources/notion/connect")
async def connect_notion(token: str) -> ConnectionResult:
    """Connect Notion workspace"""
    pass

@app.post("/api/sources/gdrive/connect")
async def connect_gdrive(
    service_account: dict,
    folders: List[str]
) -> ConnectionResult:
    """Connect Google Drive folders"""
    pass

@app.post("/api/sources/scan")
async def trigger_scan(source: Optional[str] = None):
    """Manually trigger source scan"""
    pass

# Real-time Updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    # Send processing status, queue updates, etc.
```

## Phased Implementation Rollout

### Phase 0: Foundation (Everything else depends on this)
```
├── Async Infrastructure (Celery + Redis)
├── State Machine (document lifecycle tracking)
├── Database Schema (with state management)
├── Error Handling & Retry Logic
└── Idempotent Operations
```

### Phase 1: Core Pipeline (No UI yet)
```
├── CocoIndex Integration
│   ├── Recursive chunking
│   ├── Semantic chunking
│   └── Custom chunk parameters
├── LLM Integration
│   ├── Multi-model support (GPT-4 + Gemini)
│   ├── Entity extraction
│   └── Metadata generation
├── Source Connectors
│   ├── Notion API client
│   └── Google Drive client
└── Graph Database Pipeline
    ├── Entity resolution
    └── Relationship mapping
```

### Phase 2: Minimal Working UI
```
├── FastAPI Backend
│   ├── Document CRUD endpoints
│   ├── Processing job endpoints
│   └── Approval queue endpoints
└── Next.js Frontend
    ├── Document list view
    ├── Document detail with chunks
    ├── Chunk editing interface
    ├── Entity correction UI
    └── Metadata override form
```

### Phase 3: Quality Control Layer
```
├── Preview System
│   ├── Chunk context visualization
│   ├── Hierarchical chunk display
│   └── Knowledge graph preview
├── Review Workflow
│   ├── Side-by-side model comparison
│   ├── Confidence score display
│   └── Manual approval interface
└── Testing with Real Documents
    ├── End-to-end processing
    ├── Error recovery testing
    └── Quality validation
```

### Phase 4: Production Deployment
```
├── Railway Deployment
├── Monitoring Setup (errors, job status)
├── Basic Documentation
└── Initial Document Processing
```

### Critical Path:
**Nothing works without Phase 0** → Test each phase before moving to next → **All phases must be complete before processing real documents**

## Deployment Configuration

### Railway Setup
```yaml
services:
  backend:
    source: .
    build:
      dockerfile: backend/Dockerfile
    env:
      - SUPABASE_URL
      - SUPABASE_KEY
      - QDRANT_URL
      - NEO4J_URI
      - OPENAI_API_KEY
    port: 8000
    
  frontend:
    source: ./frontend
    build:
      dockerfile: frontend/Dockerfile
    env:
      - NEXT_PUBLIC_API_URL
    port: 3000
    
  worker:
    source: .
    build:
      dockerfile: worker/Dockerfile
    env:
      - DATABASE_URL
      - REDIS_URL
```

## Security Considerations

- **Authentication**: Supabase Auth with RLS
- **API Security**: JWT tokens for all requests
- **Data Encryption**: TLS for all connections
- **Audit Logging**: Track all user actions
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Sanitize all user inputs

## Performance Targets (Adjusted for Low Volume)

### Realistic for < 100 Documents:
- **Page Load**: < 1s (acceptable for internal tool)
- **Search Latency**: < 200ms (via Qdrant)
- **Processing Time**: < 30s per document (quality over speed)
- **UI Response**: < 200ms (good enough)
- **Concurrent Users**: 2-3 (internal team only)

### Why These Are Fine:
- With < 100 docs, even linear scans are fast
- Processing can take longer since it's async
- Few users means less optimization needed

## Future Enhancements (When You Actually Need Them)

### Consider Only When:
1. **Advanced Chunking**: When > 1000 documents
2. **Collaboration**: When > 5 team members
3. **Analytics Dashboard**: When > 500 documents
4. **Custom Extractors**: When generic extraction fails
5. **API Access**: When integrating with other systems
6. **Batch Operations**: When > 50 docs/week
7. **Auto-approval**: When patterns are proven
8. **Version Control**: When compliance requires it

### What You DON'T Need (These Are Scale Features):
- Complex caching strategies
- Kubernetes/microservices
- GraphQL API
- Real-time collaboration
- Multi-tenancy
- Horizontal scaling
- Batch operations (process one doc at a time)
- Auto-approval (review everything manually)

### What You MUST Keep (These Affect Quality):
- Multiple chunking strategies
- Manual chunk editing
- Entity correction capabilities
- Custom metadata fields
- Multi-model LLM comparison
- Hierarchical chunk visualization
- Context windows for chunks
- Relationship mapping preview

## Critical Success Factors

### What Will Make or Break This Project:

1. **Async Processing**: Without Celery/Redis, the app will hang
2. **State Management**: Without proper state tracking, you'll lose documents
3. **Error Handling**: Must gracefully handle LLM/API failures
4. **Idempotency**: Prevent duplicate processing on retries

### What Can Wait:
- Beautiful UI (function over form)
- Real-time updates (polling is fine)
- Batch operations (one at a time works)
- Advanced chunking (recursive splitting is good enough)

### Architecture Decision Record

**Why Qdrant over pgvector**: 
- 2-3x better latency for vector search
- Native multi-vector support for ColPali embeddings
- Better scaling path if you grow

**Why Keep Neo4j**:
- Knowledge graphs are complex to implement
- Relationship queries are powerful for context
- Worth the complexity even for small scale

**Why Celery + Redis**:
- CocoIndex operations can take 30+ seconds
- LLM calls can timeout
- Users need responsive UI during processing

## Conclusion

This simplified design focuses on the **critical 20% that delivers 80% of value** for low-volume, high-quality document processing. By deferring complexity and focusing on robust async processing and state management, you can build a working system in 4-5 weeks instead of 8-10 weeks.

Remember: **It's easier to add features to a working simple system than to debug a complex broken one.**