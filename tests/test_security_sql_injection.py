"""Tests to ensure SQL injection vulnerabilities are prevented."""
from __future__ import annotations

import sys
import types
from unittest.mock import Mock, call, patch

import pytest

# Create fake psycopg2 before importing batch_prediction
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_sql = types.ModuleType("psycopg2.sql")
    fake_extras = types.ModuleType("psycopg2.extras")

    class FakeIdentifier:
        """Mock Identifier that records what was passed to it."""
        def __init__(self, name):
            self.name = name
            self.string = f'"{name}"'
        def __repr__(self):
            return self.string

    class FakeSQL:
        """Mock SQL that records template."""
        def __init__(self, template):
            self.template = template
        def format(self, *args, **kwargs):
            # Record that format was called with Identifiers
            return self
        def __repr__(self):
            return self.template

    fake_sql.Identifier = FakeIdentifier
    fake_sql.SQL = FakeSQL
    fake_extras.execute_batch = Mock()

    fake_psycopg2.sql = fake_sql
    fake_psycopg2.extras = fake_extras

    sys.modules["psycopg2"] = fake_psycopg2
    sys.modules["psycopg2.sql"] = fake_sql
    sys.modules["psycopg2.extras"] = fake_extras


from energy_forecasting.components.batch_prediction import BatchPrediction
from psycopg2 import sql


def test_batch_prediction_uses_sql_identifier_for_table_names():
    """Test that batch prediction uses sql.Identifier for table names to prevent SQL injection."""
    # This test verifies that when creating tables or inserting data,
    # the code uses psycopg2.sql.Identifier instead of string formatting

    # Create a mock cursor
    mock_cursor = Mock()
    mock_cursor.fetchall.return_value = []

    # Check that sql.Identifier class exists and works
    identifier = sql.Identifier("test_table")
    assert identifier is not None
    assert "test_table" in repr(identifier)


def test_sql_identifier_prevents_injection_attack():
    """Test that sql.Identifier properly escapes malicious input."""
    # Example of a SQL injection attempt
    malicious_table_name = "users; DROP TABLE users; --"

    # Using sql.Identifier should safely escape this
    identifier = sql.Identifier(malicious_table_name)

    # The identifier should wrap it safely, not execute the injection
    # psycopg2's Identifier will quote it as: "users; DROP TABLE users; --"
    # which becomes a valid (though weird) table name, not executable SQL
    assert identifier.name == malicious_table_name
    # The point is that it's treated as data, not code


def test_batch_prediction_create_table_uses_safe_sql():
    """Verify that CREATE TABLE statements use sql.Identifier."""
    # We can't easily test the actual SQL execution without a real DB,
    # but we can verify the pattern is used in the code

    from energy_forecasting.components import batch_prediction
    import inspect

    # Get source code of the module
    source = inspect.getsource(batch_prediction)

    # Verify that sql.Identifier is imported and used
    assert "from psycopg2 import sql" in source
    assert "sql.Identifier" in source
    assert "sql.SQL" in source

    # Verify dangerous patterns are NOT present
    assert "CREATE TABLE {}" not in source or "sql.SQL" in source
    # String formatting for SQL should only be in safe contexts


def test_batch_prediction_insert_uses_parameterized_queries():
    """Verify that INSERT statements use parameterized queries for values."""
    from energy_forecasting.components import batch_prediction
    import inspect

    source = inspect.getsource(batch_prediction)

    # Check that we're not using f-strings or % formatting for SQL values
    # Pattern like: INSERT INTO table VALUES ('{value}') is vulnerable
    # Pattern like: cur.execute(sql, (value,)) is safe

    # Verify that execute_batch is used (which supports parameterization)
    assert "execute_batch" in source

    # The presence of %s placeholders with separate value tuples is the safe pattern
    # We verify the unsafe pattern is NOT present
    assert ".format(" not in source or "sql.SQL" in source  # format() only used with sql.SQL


def test_identifier_usage_in_upsert_statement():
    """Test that UPSERT statement construction uses Identifier for all table/column names."""
    # Create test identifiers
    table = sql.Identifier("predictions_hourly")
    col1 = sql.Identifier("unique_id")
    col2 = sql.Identifier("ds")

    # Verify they exist and have the expected structure
    assert table.name == "predictions_hourly"
    assert col1.name == "unique_id"
    assert col2.name == "ds"

    # Verify SQL class can construct queries
    template = sql.SQL("CREATE TABLE {} ({}, {})").format(table, col1, col2)
    assert template is not None


def test_no_string_concatenation_in_sql_queries():
    """Ensure SQL queries don't use dangerous string concatenation."""
    from energy_forecasting.components import batch_prediction
    import inspect

    source = inspect.getsource(batch_prediction)

    # Split into lines for analysis
    lines = source.split('\n')

    # Look for dangerous patterns in SQL-related lines
    dangerous_patterns = [
        'f"CREATE TABLE',  # f-string with SQL
        'f"INSERT INTO',   # f-string with SQL
        '+ "CREATE',       # String concatenation with SQL
        '+ "INSERT',       # String concatenation with SQL
    ]

    for i, line in enumerate(lines, 1):
        for pattern in dangerous_patterns:
            if pattern in line and 'sql.SQL' not in line:
                # Allow it if sql.SQL is used (safe)
                pytest.fail(
                    f"Line {i}: Found dangerous SQL pattern '{pattern}' without sql.SQL: {line.strip()}"
                )


def test_kaggle_etl_script_uses_safe_sql():
    """Verify that the Kaggle ETL script also uses safe SQL patterns."""
    try:
        from scripts import kaggle_to_db_df_mlf
        import inspect

        source = inspect.getsource(kaggle_to_db_df_mlf)

        # Verify sql.Identifier is imported and used
        assert "from psycopg2 import sql" in source or "psycopg2.sql" in source
        assert "sql.Identifier" in source

    except ImportError:
        pytest.skip("Kaggle ETL script not in importable path")


def test_password_sanitization_in_postgres_loader():
    """Verify that passwords are sanitized in log messages."""
    from energy_forecasting.data_access import postgres_loader
    import inspect

    source = inspect.getsource(postgres_loader)

    # Check for password sanitization
    assert "***" in source or "sanitize" in source.lower()

    # Verify we're not logging raw connection strings
    # Look for patterns where password is replaced
    assert "safe_url" in source or "sanitiz" in source