"""Open connections to the Postgres database used by the application."""

import os

import psycopg
from pgvector.psycopg import register_vector


def get_connection() -> psycopg.Connection:
    """Connect using DATABASE_URL and teach Psycopg about pgvector values."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    connection = psycopg.connect(database_url)
    register_vector(connection)
    return connection
