-- Supabase Schema for Document Ingestion Portal
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Documents table - main document registry
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('upload', 'notion', 'gdrive', 'url')),
    source_url TEXT,
    source_id TEXT, -- External ID from source system
    file_type TEXT,
    file_size INTEGER,
    
    -- Processing state
    status TEXT NOT NULL DEFAULT 'discovered' 
        CHECK (status IN ('discovered', 'processing', 'pending_review', 'approved', 'rejected', 'ingested', 'failed')),
    processing_error TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Parsing details
    parse_tier TEXT CHECK (parse_tier IN ('balanced', 'agentic', 'agentic_plus')),
    parse_confidence FLOAT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    approved_at TIMESTAMPTZ,
    approved_by UUID,
    
    -- Versioning
    version INTEGER DEFAULT 1,
    parent_document_id UUID REFERENCES documents(id)
);

-- Chunks table - document chunks with embeddings reference
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Chunk details
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_size INTEGER,
    
    -- Chunking strategy
    chunking_strategy TEXT NOT NULL CHECK (chunking_strategy IN ('recursive', 'semantic', 'fixed', 'sentence')),
    chunk_overlap INTEGER,
    
    -- Parent-child relationship for hierarchical chunking
    parent_chunk_id UUID REFERENCES chunks(id),
    
    -- Embeddings reference (actual vectors in Qdrant)
    embedding_id TEXT, -- Qdrant point ID
    embedding_model TEXT,
    
    -- Review status
    is_edited BOOLEAN DEFAULT FALSE,
    original_text TEXT, -- Store original if edited
    edited_by UUID,
    edited_at TIMESTAMPTZ,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Entities table - extracted entities
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Entity details
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL, -- person, organization, location, concept, etc.
    confidence_score FLOAT,
    
    -- Source information
    source_chunk_id UUID REFERENCES chunks(id),
    source_model TEXT, -- Which LLM extracted this
    
    -- Review status
    is_verified BOOLEAN DEFAULT FALSE,
    is_edited BOOLEAN DEFAULT FALSE,
    original_name TEXT,
    verified_by UUID,
    verified_at TIMESTAMPTZ,
    
    -- For entity resolution
    canonical_entity_id UUID REFERENCES entities(id), -- Points to the canonical version
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Entity relationships table
CREATE TABLE IF NOT EXISTS entity_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL,
    confidence_score FLOAT,
    
    -- Source information
    source_chunk_id UUID REFERENCES chunks(id),
    source_model TEXT,
    
    -- Review status
    is_verified BOOLEAN DEFAULT FALSE,
    verified_by UUID,
    verified_at TIMESTAMPTZ,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate relationships
    UNIQUE(source_entity_id, target_entity_id, relationship_type)
);

-- Processing jobs table - track async processing
CREATE TABLE IF NOT EXISTS processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Job details
    job_type TEXT NOT NULL CHECK (job_type IN ('parse', 'chunk', 'embed', 'extract_entities', 'extract_metadata', 'full_pipeline')),
    job_status TEXT NOT NULL DEFAULT 'pending' 
        CHECK (job_status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    
    -- Celery task tracking
    celery_task_id TEXT UNIQUE,
    
    -- Progress tracking
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    current_step TEXT,
    total_steps INTEGER,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Timing
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- LLM comparisons table - store multi-model outputs
CREATE TABLE IF NOT EXISTS llm_comparisons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Comparison details
    comparison_type TEXT NOT NULL CHECK (comparison_type IN ('metadata', 'entities', 'summary', 'keywords')),
    
    -- Model outputs
    gpt4_output JSONB,
    gpt4_confidence FLOAT,
    
    gemini_output JSONB,
    gemini_confidence FLOAT,
    
    claude_output JSONB,
    claude_confidence FLOAT,
    
    -- Selected output
    selected_model TEXT,
    selected_output JSONB,
    selected_by UUID,
    selection_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    selected_at TIMESTAMPTZ
);

-- Document metadata table - structured metadata
CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Common metadata fields
    title TEXT,
    author TEXT,
    description TEXT,
    language TEXT DEFAULT 'en',
    
    -- Dates
    publication_date DATE,
    last_modified DATE,
    
    -- Classification
    category TEXT,
    subcategory TEXT,
    document_type TEXT,
    
    -- Extracted metadata
    keywords TEXT[],
    summary TEXT,
    
    -- Source of metadata
    metadata_source TEXT CHECK (metadata_source IN ('extracted', 'manual', 'source_system')),
    extraction_model TEXT,
    
    -- Review status
    is_reviewed BOOLEAN DEFAULT FALSE,
    reviewed_by UUID,
    reviewed_at TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- One metadata record per document
    UNIQUE(document_id)
);

-- Images table - extracted images from documents
CREATE TABLE IF NOT EXISTS document_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    
    -- Image details
    image_index INTEGER NOT NULL,
    page_number INTEGER,
    storage_path TEXT NOT NULL, -- Path in Supabase storage
    
    -- Image analysis
    caption TEXT,
    caption_model TEXT,
    caption_confidence FLOAT,
    
    ocr_text TEXT,
    detected_objects JSONB,
    
    -- Embeddings reference
    embedding_id TEXT, -- Qdrant point ID for ColPali embeddings
    
    -- Review status
    is_caption_edited BOOLEAN DEFAULT FALSE,
    original_caption TEXT,
    edited_by UUID,
    edited_at TIMESTAMPTZ,
    
    include_in_index BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_source_type ON documents(source_type);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_chunk_index ON chunks(document_id, chunk_index);

CREATE INDEX idx_entities_document_id ON entities(document_id);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_canonical ON entities(canonical_entity_id);

CREATE INDEX idx_processing_jobs_document_id ON processing_jobs(document_id);
CREATE INDEX idx_processing_jobs_status ON processing_jobs(job_status);
CREATE INDEX idx_processing_jobs_celery_task ON processing_jobs(celery_task_id);

CREATE INDEX idx_llm_comparisons_document_id ON llm_comparisons(document_id);
CREATE INDEX idx_document_metadata_document_id ON document_metadata(document_id);
CREATE INDEX idx_document_images_document_id ON document_images(document_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to all tables
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at BEFORE UPDATE ON chunks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_entities_updated_at BEFORE UPDATE ON entities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_processing_jobs_updated_at BEFORE UPDATE ON processing_jobs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_metadata_updated_at BEFORE UPDATE ON document_metadata 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_document_images_updated_at BEFORE UPDATE ON document_images 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Row Level Security (RLS) - Enable after configuring auth
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;

-- Grant permissions to authenticated users (uncomment when auth is set up)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'All tables created successfully!';
END $$;