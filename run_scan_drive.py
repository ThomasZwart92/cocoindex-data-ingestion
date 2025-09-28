import asyncio
from dotenv import load_dotenv
load_dotenv()
from app.connectors.google_drive_connector import GoogleDriveConnector

async def main():
    c = GoogleDriveConnector()
    docs = await c.scan_drive()
    print('scan_drive count', len(docs))
    for d in docs[:3]:
        print(d['title'], d['metadata']['mime_type'])

asyncio.run(main())
