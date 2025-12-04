"""Database utilities for connecting to PostgreSQL via SQLAlchemy."""

import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_engine() -> Engine:
    """Create a SQLAlchemy engine using the ``DATABASE_URL`` environment variable."""

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL environment variable not set. "
            "Set it to a valid PostgreSQL connection string before running."
        )
    return create_engine(db_url)


def get_session(engine: Optional[Engine] = None) -> Session:
    """Return a session bound to the provided engine (or a new one)."""

    engine = engine or get_engine()
    session_factory = sessionmaker(bind=engine)
    return session_factory()
