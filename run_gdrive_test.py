import asyncio
from dotenv import load_dotenv
load_dotenv()
from app.connectors.google_drive_connector import test_google_drive_connector
asyncio.run(test_google_drive_connector())
