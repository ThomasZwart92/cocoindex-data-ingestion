"""Fix entity extraction for NC2050 document"""
import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Supabase
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

async def fix_nc2050_entities():
    """Fix entity extraction for NC2050"""
    
    # Find NC2050 document
    docs = supabase.table('documents').select('*').ilike('name', '%NC2050%').execute()
    if not docs.data:
        print("NC2050 document not found")
        return
    
    doc = docs.data[0]
    doc_id = doc['id']
    print(f"Found document: {doc['name']} (ID: {doc_id})")
    
    # Step 1: Clear bad entities
    print("\n1. Clearing existing bad entities...")
    delete_result = supabase.table('entities').delete().eq('document_id', doc_id).execute()
    print(f"   Deleted {len(delete_result.data) if delete_result.data else 0} entities")
    
    # Step 2: Get chunks for proper entity extraction
    chunks = supabase.table('chunks').select('*').eq('document_id', doc_id).execute()
    print(f"\n2. Found {len(chunks.data)} chunks for entity extraction")
    
    # Step 3: Extract entities properly using LLM
    from app.processors.entity_extractor import EntityExtractor
    from app.models.chunk import Chunk
    
    # Convert to Chunk objects
    chunk_objects = []
    for chunk_data in chunks.data[:10]:  # Process first 10 chunks for testing
        chunk = Chunk(
            id=chunk_data['id'],
            document_id=doc_id,
            chunk_index=chunk_data['chunk_index'],
            chunk_text=chunk_data['chunk_text'],
            chunk_size=chunk_data['chunk_size'],
            chunking_strategy=chunk_data.get('chunking_strategy', 'three_tier'),
            metadata=chunk_data.get('metadata', {})
        )
        chunk_objects.append(chunk)
    
    print(f"\n3. Extracting entities from {len(chunk_objects)} chunks using LLM...")
    extractor = EntityExtractor()
    entities, relationships = extractor.extract(chunk_objects, doc_id)
    
    print(f"   Extracted {len(entities)} entities")
    
    # Step 4: Save proper entities
    if entities:
        entities_to_insert = []
        for entity in entities:
            entity_record = {
                "id": entity.id,
                "document_id": doc_id,
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                "confidence_score": entity.confidence,  # Changed from 'confidence' to 'confidence_score'
                "metadata": entity.metadata,
                "created_at": datetime.utcnow().isoformat()
                # Note: 'updated_at' doesn't exist in the entities table schema
            }
            entities_to_insert.append(entity_record)
        
        # Insert entities
        result = supabase.table("entities").insert(entities_to_insert).execute()
        print(f"\n4. Saved {len(entities_to_insert)} proper entities to database")
        
        # Display the entities
        print("\nExtracted entities:")
        for entity in entities[:10]:  # Show first 10
            print(f"  - {entity.entity_name} ({entity.entity_type}) - Confidence: {entity.confidence:.2f}")
    else:
        print("\n4. No entities were extracted (document may not contain identifiable entities)")
    
    # Step 5: Update document metadata
    current_metadata = doc.get('metadata', {}) or {}
    updated_metadata = {
        **current_metadata,
        "entity_extraction_fixed": True,
        "entity_extraction_date": datetime.utcnow().isoformat(),
        "entity_count": len(entities) if entities else 0
    }
    
    supabase.table("documents").update({
        "metadata": updated_metadata
    }).eq("id", doc_id).execute()
    
    print(f"\n5. Updated document metadata")
    print("\n✅ Entity extraction fixed successfully!")

if __name__ == "__main__":
    asyncio.run(fix_nc2050_entities())