import os
from dotenv import load_dotenv
load_dotenv()
print('Has COHERE_API_KEY:', bool(os.getenv('COHERE_API_KEY')))
print('COHERE_RERANK_MODEL:', os.getenv('COHERE_RERANK_MODEL','(default)'))
