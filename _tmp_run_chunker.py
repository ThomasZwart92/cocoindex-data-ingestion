import json
from app.services.supabase_service import SupabaseService
from app.processors.two_tier_chunker import TwoTierChunker

service = SupabaseService()
document_id = '21a57af4-f27f-4495-a36a-a9467454e204'
doc = service.get_document(document_id)
chunker = TwoTierChunker()

import asyncio

async def run():
    chunks = await chunker.process_document(
        document_id=document_id,
        content=doc.content or '',
        title=doc.name,
        metadata=doc.metadata
    )
    print('chunks length', len(chunks))
    for c in chunks[:5]:
        print(c)

asyncio.run(run())
