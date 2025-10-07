from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings


_engine = create_engine(settings.database_url, echo=False, future=True)

# Prevent repeated metadata registration during Streamlit reruns
_db_initialized = False


def init_db() -> None:
    """Create all database tables; call during app bootstrap."""

    global _db_initialized
    if _db_initialized:
        return

    SQLModel.metadata.create_all(_engine)
    _db_initialized = True


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for database operations."""

    session = Session(_engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine():
    """Return the shared SQLModel engine instance."""
    return _engine
