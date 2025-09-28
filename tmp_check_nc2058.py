from app.services.supabase_service import SupabaseService
import json

svc = SupabaseService()

doc_name = "NC2058 - Identification & resolution"

doc_res = (
    svc.client
    .table('documents')
    .select('id,name,status,created_at,updated_at')
    .eq('name', doc_name)
    .execute()
)
data = doc_res.data or []
if not data:
    print(json.dumps({"error": "Document not found", "name": doc_name}))
    raise SystemExit(0)

document = data[0]
print(json.dumps({"document": document}, indent=2))

doc_id = document['id']

mentions_res = (
    svc.client
    .table('entity_mentions')
    .select('*')
    .eq('document_id', doc_id)
    .execute()
)
mentions = mentions_res.data or []
print(json.dumps({"mention_count": len(mentions)}, indent=2))

with_canonical = [m for m in mentions if m.get('canonical_entity_id')]
without_canonical = [m for m in mentions if not m.get('canonical_entity_id')]
print(json.dumps({
    "canonical_mentions": len(with_canonical),
    "no_canonical": len(without_canonical)
}, indent=2))

if with_canonical:
    canonical_ids = sorted({m['canonical_entity_id'] for m in with_canonical if m.get('canonical_entity_id')})
    print(json.dumps({"canonical_ids": canonical_ids}, indent=2))

    canonicals_res = (
        svc.client
        .table('canonical_entities')
        .select('id,name,type,metadata,mention_count,document_count,relationship_count,quality_score')
        .in_('id', canonical_ids)
        .execute()
    )
    canonicals = canonicals_res.data or []
    print(json.dumps({"canonical_entities": canonicals}, indent=2))

relationships_res = (
    svc.client
    .table('canonical_relationships')
    .select('id,relationship_type,source_entity_id,target_entity_id,metadata')
    .eq('metadata->>document_id', doc_id)
    .execute()
)
relationships = relationships_res.data or []
print(json.dumps({"relationship_count": len(relationships)}, indent=2))
if relationships:
    print(json.dumps({"relationships": relationships}, indent=2))
