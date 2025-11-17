from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
else:
    ASYNC_DATABASE_URL = DATABASE_URL

engine = create_async_engine(ASYNC_DATABASE_URL)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()

async def init_db():
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables created successfully!")