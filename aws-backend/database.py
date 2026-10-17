import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, Text, DateTime, func

WRITER_DB_URL = os.getenv(
    "RDS_WRITER_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/interviewdb"
)
READER_DB_URL = os.getenv("RDS_READER_URL", WRITER_DB_URL)

writer_engine = create_async_engine(WRITER_DB_URL, pool_size=10, max_overflow=10, pool_pre_ping=True)
reader_engine = create_async_engine(READER_DB_URL, pool_size=20, max_overflow=10, pool_pre_ping=True)

WriteSessionLocal = async_sessionmaker(bind=writer_engine, class_=AsyncSession, expire_on_commit=False)
ReadSessionLocal = async_sessionmaker(bind=reader_engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

class InterviewSessionDB(Base):
    __tablename__ = "interview_sessions"
    session_id = Column(String(64), primary_key=True, index=True)
    questions_json = Column(Text, nullable=False)
    transcript_json = Column(Text, nullable=False, default="[]")
    current_index = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

async def get_write_db():
    async with WriteSessionLocal() as session:
        yield session

async def get_read_db():
    async with ReadSessionLocal() as session:
        yield session
