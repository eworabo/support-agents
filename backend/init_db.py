import asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from models import Base  # Your models.py
import os
from dotenv import load_dotenv

# Load .env for DATABASE_URL
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env")

async def init_db():
    engine = create_async_engine(DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created successfully!")

asyncio.run(init_db())