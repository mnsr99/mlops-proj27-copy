import os
from contextlib import contextmanager
from typing import Any

import psycopg2


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def get_db_config() -> dict[str, Any]:
    return {
        "host": _env("DB_HOST", "postgres"),
        "port": int(_env("DB_PORT", "5432")),
        "dbname": _env("DB_NAME", "jitsi_mlops"),
        "user": _env("DB_USER", "user"),
        "password": _env("DB_PASSWORD", "jitsi_postgres"),
    }


def get_conn():
    return psycopg2.connect(**get_db_config())


@contextmanager
def db_cursor(commit: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    try:
        yield conn, cur
        if commit:
            conn.commit()
    finally:
        cur.close()
        conn.close()
