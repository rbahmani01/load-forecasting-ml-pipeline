from __future__ import annotations

import os
import time

import psycopg2
from sqlalchemy import create_engine
import pandas as pd

from energy_forecasting.entity.config_entity import DatabaseConfig
from energy_forecasting.exception import EnergyException
from energy_forecasting.logger import logger


def build_postgres_url(cfg: DatabaseConfig) -> str:
    """
    Build a SQLAlchemy connection URL for Postgres.

    Environment variables (if set) override cfg.
    - Local venv: it uses cfg.
    - Docker: we set ENERGY_DB_HOST=energy-db.
    """
    host = os.getenv("ENERGY_DB_HOST", cfg.host)
    port = os.getenv("ENERGY_DB_PORT", str(cfg.port))
    user = os.getenv("ENERGY_DB_USER", cfg.user)
    password = os.getenv("ENERGY_DB_PASSWORD", cfg.password)
    db_name = os.getenv("ENERGY_DB_NAME", cfg.db_name)

    conn_str = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
    # Log sanitized URL without password
    safe_url = f"postgresql+psycopg2://{user}:***@{host}:{port}/{db_name}"
    logger.info("Using Postgres URL: %s", safe_url)
    return conn_str


def _retry_on_db_error(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retry a database operation with exponential backoff.
    
    Args:
        func: Callable that performs the DB operation
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds (exponential backoff)
    
    Returns:
        Result of func if successful
        
    Raises:
        EnergyException: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                logger.warning(
                    "DB connection failed (attempt %d/%d), retrying in %.1f seconds: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    str(e),
                )
                time.sleep(delay)
            else:
                logger.error(
                    "DB connection failed after %d attempts: %s",
                    max_retries,
                    str(e),
                )
        except Exception as e:
            # Non-retryable errors fail immediately
            raise EnergyException(f"Database error: {e}") from e
    
    # All retries exhausted
    raise EnergyException(
        f"Failed to connect to database after {max_retries} attempts: {last_error}"
    ) from last_error


def load_recent_table_from_db(db_config: DatabaseConfig) -> pd.DataFrame:
    conn_str = build_postgres_url(db_config)

    # For logging, resolve what we actually use
    host = os.getenv("ENERGY_DB_HOST", db_config.host)
    port = os.getenv("ENERGY_DB_PORT", str(db_config.port))
    db_name = os.getenv("ENERGY_DB_NAME", db_config.db_name)

    logger.info(
        "Connecting to Postgres %s:%s db=%s, table=%s (recent window: %d hours)",
        host,
        port,
        db_name,
        db_config.table,
        db_config.hours_history,
    )

    engine = create_engine(conn_str)
    
    # Safe SQL with identifier quoting - use double quotes for table names
    safe_table = f'"{db_config.table}"'
    hours = db_config.hours_history
    query = f"""SELECT *
        FROM {safe_table}
        WHERE ds >= (
            SELECT MAX(ds) FROM {safe_table}
        ) - INTERVAL '{hours} hour'
        ORDER BY ds"""

    def _load():
        conn = None
        try:
            conn = engine.raw_connection()
            df = pd.read_sql(query, conn)
            if df.empty:
                logger.warning(
                    "Loaded 0 rows from DB table '%s' (last %d hours).",
                    db_config.table,
                    db_config.hours_history,
                )
            else:
                logger.info(
                    "Loaded %d rows from DB (last %d hours).",
                    len(df),
                    db_config.hours_history,
                )
            return df
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug("Error closing connection: %s", e)

    return _retry_on_db_error(_load, max_retries=3, initial_delay=1.0)


def load_full_table_from_db(db_config: DatabaseConfig) -> pd.DataFrame:
    conn_str = build_postgres_url(db_config)

    host = os.getenv("ENERGY_DB_HOST", db_config.host)
    port = os.getenv("ENERGY_DB_PORT", str(db_config.port))
    db_name = os.getenv("ENERGY_DB_NAME", db_config.db_name)

    logger.info(
        "Connecting to Postgres %s:%s db=%s, table=%s (full table load)",
        host,
        port,
        db_name,
        db_config.table,
    )

    engine = create_engine(conn_str)
    
    # Safe SQL with identifier quoting - use double quotes for table names
    safe_table = f'"{db_config.table}"'
    query = f"SELECT * FROM {safe_table}"

    def _load():
        conn = None
        try:
            conn = engine.raw_connection()
            df = pd.read_sql(query, conn)
            logger.info("Loaded %d rows from DB (full table).", len(df))
            return df
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception as e:
                    logger.debug("Error closing connection: %s", e)

    return _retry_on_db_error(_load, max_retries=3, initial_delay=1.0)
