import requests
import json

document_id = "9ca98da2-3332-4891-9510-9c20aebd3ea9"
base = "http://localhost:8005"
entities = requests.get(f"{base}/api/documents/{document_id}/entities").json()
relationships = requests.get(f"{base}/api/documents/{document_id}/relationships").json()
print(json.dumps({
    "entity_count": len(entities),
    "relationship_count": len(relationships)
}, indent=2))
print(json.dumps(relationships, indent=2))
