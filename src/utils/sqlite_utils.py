import sqlite3
from typing import Any, Dict

_DB_FILE = "bud_hardwares.db"


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a SQLite connection using the given path or default."""
    path = db_path or _DB_FILE
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_db(conn)
    return conn


def initialize_db(conn: sqlite3.Connection) -> None:
    """Create the hardware table if it doesn't exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hardware (
            name TEXT PRIMARY KEY,
            flops INTEGER,
            memory_size INTEGER,
            memory_bw INTEGER,
            icn INTEGER,
            icn_ll REAL,
            real_values BOOLEAN
        )
        """
    )
    conn.commit()


def fetch_all(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    cur = conn.execute("SELECT * FROM hardware")
    rows = cur.fetchall()
    return {
        row["name"]: {
            "Flops": row["flops"],
            "Memory_size": row["memory_size"],
            "Memory_BW": row["memory_bw"],
            "ICN": row["icn"],
            "ICN_LL": row["icn_ll"],
            "real_values": bool(row["real_values"]),
        }
        for row in rows
    }


def upsert(conn: sqlite3.Connection, name: str, config: Dict[str, Any]) -> None:
    data = {
        "name": name,
        "Flops": config.get("Flops"),
        "Memory_size": config.get("Memory_size"),
        "Memory_BW": config.get("Memory_BW"),
        "ICN": config.get("ICN"),
        "ICN_LL": config.get("ICN_LL"),
        "real_values": config.get("real_values", True),
    }
    conn.execute(
        """
        INSERT INTO hardware (name, flops, memory_size, memory_bw, icn, icn_ll, real_values)
        VALUES (:name, :Flops, :Memory_size, :Memory_BW, :ICN, :ICN_LL, :real_values)
        ON CONFLICT(name) DO UPDATE SET
            flops=excluded.flops,
            memory_size=excluded.memory_size,
            memory_bw=excluded.memory_bw,
            icn=excluded.icn,
            icn_ll=excluded.icn_ll,
            real_values=excluded.real_values
        """,
        data,
    )
    conn.commit()


def delete(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("DELETE FROM hardware WHERE name=?", (name,))
    conn.commit()
