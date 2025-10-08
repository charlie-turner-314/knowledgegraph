from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings



_engine = create_engine(settings.database_url, echo=False, future=True)

import logging
from sqlalchemy import inspect

def init_db() -> None:
    """Idempotently ensure all tables exist and match the model schema. If missing, create. If mismatched, error."""
    inspector = inspect(_engine)
    required_tables = set(SQLModel.metadata.tables.keys())
    existing_tables = set(inspector.get_table_names())

    missing = required_tables - existing_tables
    if missing:
        logging.info(f"Creating missing tables: {sorted(missing)}")
        SQLModel.metadata.create_all(_engine)
        return

    # Check schema for each table
    for table_name, model_table in SQLModel.metadata.tables.items():
        db_columns = {col['name'] for col in inspector.get_columns(table_name)}
        model_columns = set(model_table.columns.keys())
        if db_columns != model_columns:
            logging.error(f"Schema mismatch for table '{table_name}': DB columns {sorted(db_columns)} vs Model columns {sorted(model_columns)}")
            raise RuntimeError(f"Database schema mismatch for table '{table_name}'. Please recreate the database.")


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
