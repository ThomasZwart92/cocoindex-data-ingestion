import asyncio
from app.services.llamaparse_service import LlamaParseService

async def t():
    s = LlamaParseService()
    try:
        r = await s.parse_from_url('https://arxiv.org/pdf/1810.04805.pdf')
        print('ok', bool(r.get('markdown')))
    except Exception as e:
        print('err', e)

asyncio.run(t())
