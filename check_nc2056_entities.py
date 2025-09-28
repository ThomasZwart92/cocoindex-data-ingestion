"""Check entity extraction for NC2056 document."""

from app.services.supabase_service import SupabaseService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

supabase = SupabaseService()

# Get document ID - use name instead of title
doc_res = supabase.client.table('documents').select('id, name').eq('name', 'NC2056 - Black Screen & Config File corruption').execute()
if not doc_res.data:
    print("Document NC2056 not found")
    exit(1)

doc_id = doc_res.data[0]['id']
print(f'Document ID: {doc_id}')

# Get all entity mentions
mentions_res = supabase.client.table('entity_mentions').select('*').eq('document_id', doc_id).execute()
mentions = mentions_res.data or []
print(f'\nFound {len(mentions)} total mentions')

# Analyze canonical assignment
with_canonical = [m for m in mentions if m.get('canonical_entity_id')]
without_canonical = [m for m in mentions if not m.get('canonical_entity_id')]

print(f'  - With canonical ID: {len(with_canonical)}')
print(f'  - Without canonical ID: {len(without_canonical)}')

if without_canonical:
    print('\nMentions WITHOUT canonical IDs (first 10):')
    for m in without_canonical[:10]:
        print(f'  - "{m["text"]}" (type: {m.get("type", "None")})')

# Check canonical entities
if with_canonical:
    # Get unique canonical IDs
    canonical_ids = list(set(m['canonical_entity_id'] for m in with_canonical if m.get('canonical_entity_id')))

    print(f'\nUnique canonical entities referenced: {len(canonical_ids)}')

    # Fetch canonical entities
    canonicals_res = supabase.client.table('canonical_entities').select('*').in_('id', canonical_ids[:5]).execute()
    if canonicals_res.data:
        print('\nSample canonical entities:')
        for ce in canonicals_res.data:
            print(f'  - {ce["name"]} (type: {ce["type"]}, id: {ce["id"][:8]}...)')

# Check relationships
rel_res = supabase.client.table('canonical_relationships').select('*').eq('metadata->>document_id', doc_id).execute()
relationships = rel_res.data or []
print(f'\nTotal relationships for document: {len(relationships)}')

if relationships:
    # Check if source/target IDs exist in canonical_entities
    rel_entity_ids = set()
    for r in relationships:
        rel_entity_ids.add(r['source_entity_id'])
        rel_entity_ids.add(r['target_entity_id'])

    print(f'Unique entities referenced in relationships: {len(rel_entity_ids)}')

    # Check which ones are valid
    valid_entities_res = supabase.client.table('canonical_entities').select('id').in_('id', list(rel_entity_ids)).execute()
    valid_ids = set(e['id'] for e in (valid_entities_res.data or []))

    invalid_ids = rel_entity_ids - valid_ids
    if invalid_ids:
        print(f'\nWARNING: {len(invalid_ids)} entity IDs in relationships are NOT in canonical_entities table!')
        print('These are likely entity mention IDs being used instead of canonical entity IDs')
        for invalid_id in list(invalid_ids)[:10]:
            print(f'  - {invalid_id}')
            # Check if it's a mention ID
            mention_check = supabase.client.table('entity_mentions').select('text, type, canonical_entity_id').eq('id', invalid_id).execute()
            if mention_check.data:
                m = mention_check.data[0]
                canonical_str = m.get("canonical_entity_id", "None")
                if canonical_str == "None" or not canonical_str:
                    canonical_str = "NO CANONICAL ID!"
                else:
                    canonical_str = canonical_str[:8] + "..."
                print(f'    ^ This IS a mention: "{m["text"]}" (type: {m.get("type")}) -> canonical: {canonical_str}')