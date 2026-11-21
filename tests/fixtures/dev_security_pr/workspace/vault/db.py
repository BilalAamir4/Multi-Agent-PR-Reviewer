"""User records in SQLite."""

import sqlite3


def connect(path):
    return sqlite3.connect(path)


def create_tables(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)"
    )


def add_user(conn, name, email):
    conn.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    conn.commit()


def find_user(conn, name):
    """The user called ``name`` as (id, name, email), or None."""
    return conn.execute(
        "SELECT id, name, email FROM users WHERE name = ?", (name,)
    ).fetchone()


def search_users(conn, term, order_by="name"):
    """Users whose name contains ``term``, sorted by the column ``order_by``."""
    query = f"SELECT id, name, email FROM users WHERE name LIKE '%{term}%' ORDER BY {order_by}"
    return conn.execute(query).fetchall()
