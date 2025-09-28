import requests
import json

document_id = "9ca98da2-3332-4891-9510-9c20aebd3ea9"
base = "http://localhost:8005"
resp = requests.post(f"{base}/api/documents/{document_id}/process", json={"force_reprocess": True})
print(resp.status_code)
print(resp.text[:200])
