-- Schema updates for Contextual Retrieval implementation
-- Based on Anthropic's approach: https://www.anthropic.com/news/contextual-retrieval

-- Add columns to chunks table for contextual retrieval
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS contextual_summary TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS contextualized_text TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_level TEXT CHECK (chunk_level IN ('page', 'paragraph', 'semantic'));
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_context TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS position_in_parent INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS bm25_tokens TEXT[];  -- For BM25 search

-- Add index for BM25 search
CREATE INDEX IF NOT EXISTS idx_chunks_bm25_tokens ON chunks USING GIN(bm25_tokens);

-- Add index for hierarchical queries
CREATE INDEX IF NOT EXISTS idx_chunks_parent_child ON chunks(parent_chunk_id, position_in_parent);
CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(chunk_level);

-- Create table for storing search metrics
CREATE TABLE IF NOT EXISTS search_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query TEXT NOT NULL,
    search_type TEXT CHECK (search_type IN ('semantic', 'bm25', 'hybrid', 'hybrid_reranked')),
    chunks_retrieved INTEGER,
    chunks_after_rerank INTEGER,
    relevance_score FLOAT,
    response_time_ms INTEGER,
    user_feedback TEXT CHECK (user_feedback IN ('helpful', 'not_helpful', 'partially_helpful')),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create table for BM25 document statistics
CREATE TABLE IF NOT EXISTS bm25_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    total_chunks INTEGER,
    avg_chunk_length FLOAT,
    vocabulary_size INTEGER,
    idf_scores JSONB,  -- Inverse document frequency scores
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create view for hierarchical chunk traversal
CREATE OR REPLACE VIEW chunk_hierarchy AS
WITH RECURSIVE chunk_tree AS (
    -- Base case: top-level chunks
    SELECT 
        c.id,
        c.document_id,
        c.chunk_text,
        c.contextual_summary,
        c.contextualized_text,
        c.chunk_level,
        c.parent_chunk_id,
        c.position_in_parent,
        0 as depth,
        ARRAY[c.id] as path
    FROM chunks c
    WHERE c.parent_chunk_id IS NULL
    
    UNION ALL
    
    -- Recursive case: child chunks
    SELECT 
        c.id,
        c.document_id,
        c.chunk_text,
        c.contextual_summary,
        c.contextualized_text,
        c.chunk_level,
        c.parent_chunk_id,
        c.position_in_parent,
        ct.depth + 1,
        ct.path || c.id
    FROM chunks c
    INNER JOIN chunk_tree ct ON c.parent_chunk_id = ct.id
)
SELECT * FROM chunk_tree;

-- Function to calculate BM25 scores
CREATE OR REPLACE FUNCTION calculate_bm25_score(
    query_tokens TEXT[],
    doc_tokens TEXT[],
    avg_doc_length FLOAT,
    doc_frequency JSONB,
    k1 FLOAT DEFAULT 1.2,
    b FLOAT DEFAULT 0.75
) RETURNS FLOAT AS $$
DECLARE
    score FLOAT := 0;
    token TEXT;
    tf FLOAT;
    idf FLOAT;
    doc_length INTEGER;
BEGIN
    doc_length := array_length(doc_tokens, 1);
    
    FOREACH token IN ARRAY query_tokens LOOP
        -- Calculate term frequency
        tf := array_length(array_positions(doc_tokens, token), 1);
        IF tf IS NULL THEN
            tf := 0;
        END IF;
        
        -- Get IDF from doc_frequency
        idf := COALESCE((doc_frequency->token)::FLOAT, 0);
        
        -- Calculate BM25 component for this term
        score := score + idf * (tf * (k1 + 1)) / 
                 (tf + k1 * (1 - b + b * doc_length / avg_doc_length));
    END LOOP;
    
    RETURN score;
END;
$$ LANGUAGE plpgsql;

-- Add comment explaining the schema
COMMENT ON TABLE chunks IS 'Stores document chunks with contextual enrichment for improved retrieval. Based on Anthropic contextual retrieval approach.';
COMMENT ON COLUMN chunks.contextual_summary IS 'LLM-generated 50-100 token context explaining what the chunk is about';
COMMENT ON COLUMN chunks.contextualized_text IS 'Combination of contextual_summary + chunk_text for improved search';
COMMENT ON COLUMN chunks.chunk_level IS 'Hierarchical level: page (1200 tokens), paragraph (400 tokens), or semantic';
COMMENT ON COLUMN chunks.bm25_tokens IS 'Tokenized text for BM25 lexical search';