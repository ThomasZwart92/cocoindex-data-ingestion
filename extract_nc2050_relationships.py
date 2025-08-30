"""Extract and save relationships for NC2050 document"""
import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import logging
import json
from typing import List, Dict, Any
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Supabase
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

# Technical documentation relationship types (matching frontend)
RELATIONSHIP_TYPES = [
    # Technical
    "COMPONENT_OF",
    "CONNECTS_TO", 
    "DEPENDS_ON",
    "REPLACES",
    "COMPATIBLE_WITH",
    "TROUBLESHOOTS",
    "CAUSES",
    "PREVENTS",
    "REQUIRES",
    "RESOLVES",
    # Documentation
    "DEFINES",
    "DOCUMENTS",
    "REFERENCES",
    "TARGETS",
    # Business
    "RESPONSIBLE_FOR",
    "SERVES",
    "IMPACTS",
    # Flexible
    "RELATES_TO"
]

async def extract_relationships_for_nc2050():
    """Extract relationships between entities for NC2050"""
    
    # Find NC2050 document
    docs = supabase.table('documents').select('*').ilike('name', '%NC2050%').execute()
    if not docs.data:
        print("NC2050 document not found")
        return
    
    doc = docs.data[0]
    doc_id = doc['id']
    print(f"Found document: {doc['name']} (ID: {doc_id})")
    
    # Get entities
    entities = supabase.table('entities').select('*').eq('document_id', doc_id).execute()
    print(f"Found {len(entities.data)} entities")
    
    if len(entities.data) < 2:
        print("Need at least 2 entities to create relationships")
        return
    
    # Get chunks to understand context
    chunks = supabase.table('chunks').select('*').eq('document_id', doc_id).limit(5).execute()
    
    # Combine chunk texts for context
    context_text = "\n".join([chunk['chunk_text'][:500] for chunk in chunks.data])
    
    # Use LLM to identify relationships
    from app.services.llm_service import LLMService, LLMProvider
    
    llm_service = LLMService()
    
    # Create entity lookup
    entity_map = {e['entity_name'].lower(): e for e in entities.data}
    entity_names = [e['entity_name'] for e in entities.data]
    
    # More specific prompt for technical documentation
    relationship_types_str = ', '.join(RELATIONSHIP_TYPES)
    prompt = f"""You are analyzing technical documentation about display issues and repairs.

Document context:
{context_text[:1500]}

Entities found in the document:
{', '.join(entity_names)}

Based on the context, identify relationships between these entities. Focus on technical/causal relationships.

Return ONLY a valid JSON array with relationships. Each relationship should have:
- source: exact entity name from the list
- target: exact entity name from the list  
- type: one of [{relationship_types_str}]
- confidence: 0.5 to 1.0 based on how clearly the relationship is stated
- context: brief explanation (max 10 words)

Relationship type meanings:
- CAUSES: source causes target (problem/issue)
- RESOLVES: source fixes/solves target
- PREVENTS: source prevents target from occurring
- REQUIRES: source needs target to function
- CONNECTS_TO: physical or logical connection
- COMPONENT_OF: source is part of target
- COMPATIBLE_WITH: can be used together
- IMPACTS: source affects target
- TROUBLESHOOTS: source helps diagnose target

Example format:
[
  {{
    "source": "isopropyl alcohol",
    "target": "corrosion",
    "type": "RESOLVES",
    "confidence": 0.9,
    "context": "Cleans corrosion from connector"
  }}
]

Only include relationships that are clearly implied or stated in the context.
Return ONLY the JSON array, no other text."""

    try:
        response = await llm_service.call_llm(
            prompt=prompt,
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=1000
        )
        
        # Clean up response to get just JSON
        content = response.content.strip()
        
        # Remove markdown if present
        if content.startswith("```json"):
            content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        elif content.startswith("```"):
            content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        
        # Parse JSON
        try:
            relationships = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                relationships = json.loads(json_match.group())
            else:
                print(f"Failed to parse JSON from response: {content[:200]}")
                relationships = []
        
        print(f"\nExtracted {len(relationships)} relationships")
        
        # Save relationships to database
        if relationships:
            relationships_to_insert = []
            
            for rel in relationships:
                # Find source and target entities
                source_entity = entity_map.get(rel['source'].lower())
                target_entity = entity_map.get(rel['target'].lower())
                
                if not source_entity or not target_entity:
                    print(f"Skipping relationship: {rel['source']} -> {rel['target']} (entity not found)")
                    continue
                
                if source_entity['id'] == target_entity['id']:
                    print(f"Skipping self-relationship: {rel['source']}")
                    continue
                
                relationship_record = {
                    "id": str(uuid.uuid4()),
                    "source_entity_id": source_entity['id'],
                    "target_entity_id": target_entity['id'],
                    "relationship_type": rel['type'],
                    "confidence_score": rel.get('confidence', 0.7),
                    "metadata": {
                        "context": rel.get('context', ''),
                        "extraction_method": "llm",
                        "extraction_date": datetime.utcnow().isoformat()
                    },
                    "created_at": datetime.utcnow().isoformat()
                }
                relationships_to_insert.append(relationship_record)
                
                print(f"  - {rel['source']} --[{rel['type']}]--> {rel['target']} ({rel.get('confidence', 0.7)*100:.0f}%)")
            
            if relationships_to_insert:
                # Clear existing relationships for these entities
                for entity in entities.data:
                    supabase.table('entity_relationships').delete().eq('source_entity_id', entity['id']).execute()
                
                # Insert new relationships
                result = supabase.table('entity_relationships').insert(relationships_to_insert).execute()
                print(f"\nSaved {len(relationships_to_insert)} relationships to database")
            else:
                print("No valid relationships to save")
        else:
            print("No relationships extracted")
            
    except Exception as e:
        logger.error(f"Error extracting relationships: {e}", exc_info=True)
        print(f"\nFailed to extract relationships: {e}")
        print("Please check LLM API configuration and try again.")

if __name__ == "__main__":
    asyncio.run(extract_relationships_for_nc2050())