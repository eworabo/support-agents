import asyncio
import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") or 'postgresql+asyncpg://postgres:postgres@localhost/support_agents_db'  # Fallback

# Strip '+asyncpg' for asyncpg.connect
if '+asyncpg' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace('+asyncpg', '')

async def test_conn():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        print("Connected successfully!")
        await conn.close()
    except Exception as e:
        print(f"Connection failed: {str(e)}")

asyncio.run(test_conn())