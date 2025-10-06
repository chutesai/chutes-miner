"""
Database setup/config/funcs.
"""

import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, Session
from chutes_miner.api.config import settings


engine = create_async_engine(settings.sqlalchemy, echo=settings.debug)

SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# Sync engine for synchronous operations
sync_engine = create_engine(
    settings.sqlalchemy.replace("+asyncpg", ""),
    echo=settings.debug,
    pool_size=5,
    max_overflow=5,
    pool_timeout=30,
)
SyncSessionLocal = sessionmaker(bind=sync_engine, class_=Session, expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session():
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    with SyncSessionLocal() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def generate_uuid():
    """
    Helper for uuid generation.
    """
    return str(uuid.uuid4())
