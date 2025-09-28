from dotenv import load_dotenv
load_dotenv()
from app.services.reranker_service import RerankerService
from app.services.search_service import SearchResult

# Build simple candidates with varying relevance
candidates = [
    SearchResult(id='1', score=0.2, source='hybrid', title='Doc 1', content='Pump troubleshooting and error E501 resolution steps', metadata={}),
    SearchResult(id='2', score=0.5, source='hybrid', title='Doc 2', content='Annual report for marketing team', metadata={}),
    SearchResult(id='3', score=0.4, source='hybrid', title='Doc 3', content='Temperature sensor and control board calibration', metadata={}),
]

query = 'error E501 pump'
rr = RerankerService()
print('use_cohere =', rr.use_cohere)
print('model =', rr.model)
import asyncio
async def main():
    out = await rr.rerank(query, candidates, top_k=3)
    print('reranked ids by score:', [c.id for c in out])
    print('scores:', [round(c.score,4) for c in out])
asyncio.run(main())
