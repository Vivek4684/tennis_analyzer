"""
SQLite database helpers for Tennis Ball In/Out Detector.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", "tennis_calls.db"))


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS calls (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at    TEXT    NOT NULL,
                result_in_pct REAL    NOT NULL,
                result_out_pct REAL   NOT NULL,
                confidence    TEXT    NOT NULL,
                explanation   TEXT    NOT NULL,
                num_frames    INTEGER NOT NULL
            );
        """)


def save_call(result_in_pct: float, result_out_pct: float, confidence: str,
              explanation: str, num_frames: int) -> int:
    """Save a ball in/out call result. Returns the new call id."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO calls (created_at, result_in_pct, result_out_pct, confidence, explanation, num_frames) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (now, result_in_pct, result_out_pct, confidence, explanation, num_frames),
        )
        return cur.lastrowid


def get_all_calls() -> list:
    """Return all past calls ordered by most recent first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM calls ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "created_at": row["created_at"],
            "result_in_pct": row["result_in_pct"],
            "result_out_pct": row["result_out_pct"],
            "confidence": row["confidence"],
            "explanation": row["explanation"],
            "num_frames": row["num_frames"],
        }
        for row in rows
    ]


def delete_call(call_id: int):
    """Delete a call by id."""
    with get_conn() as conn:
        conn.execute("DELETE FROM calls WHERE id = ?", (call_id,))
