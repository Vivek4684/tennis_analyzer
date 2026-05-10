"""
SQLite database helpers for Tennis Match Analyzer.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.environ.get("DB_PATH", "tennis_matches.db"))


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT    NOT NULL,
                video_filename TEXT NOT NULL,
                team1_score REAL,
                team2_score REAL,
                match_summary TEXT
            );

            CREATE TABLE IF NOT EXISTS players (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id     INTEGER NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                name         TEXT    NOT NULL,
                position     TEXT    NOT NULL,
                overall_rating REAL,
                strengths    TEXT,
                improvements TEXT
            );
        """)


def save_match(video_filename: str, analysis: dict) -> int:
    """Persist a completed analysis. Returns the new match id."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO matches (created_at, video_filename, team1_score, team2_score, match_summary) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                now,
                video_filename,
                analysis.get("team1_score"),
                analysis.get("team2_score"),
                analysis.get("match_summary", ""),
            ),
        )
        match_id = cur.lastrowid
        for p in analysis.get("players", []):
            conn.execute(
                "INSERT INTO players (match_id, name, position, overall_rating, strengths, improvements) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    match_id,
                    p["name"],
                    p["position"],
                    p.get("overall_rating"),
                    json.dumps(p.get("strengths", [])),
                    json.dumps(p.get("improvements", [])),
                ),
            )
    return match_id


def _row_to_match(row, players):
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "video_filename": row["video_filename"],
        "team1_score": row["team1_score"],
        "team2_score": row["team2_score"],
        "match_summary": row["match_summary"],
        "players": [
            {
                "id": p["id"],
                "name": p["name"],
                "position": p["position"],
                "overall_rating": p["overall_rating"],
                "strengths": json.loads(p["strengths"] or "[]"),
                "improvements": json.loads(p["improvements"] or "[]"),
            }
            for p in players
        ],
    }


def get_all_matches() -> list:
    with get_conn() as conn:
        matches = conn.execute(
            "SELECT * FROM matches ORDER BY created_at DESC"
        ).fetchall()
        result = []
        for m in matches:
            players = conn.execute(
                "SELECT * FROM players WHERE match_id = ? ORDER BY position",
                (m["id"],),
            ).fetchall()
            result.append(_row_to_match(m, players))
    return result


def get_match(match_id: int) -> dict | None:
    with get_conn() as conn:
        m = conn.execute(
            "SELECT * FROM matches WHERE id = ?", (match_id,)
        ).fetchone()
        if not m:
            return None
        players = conn.execute(
            "SELECT * FROM players WHERE match_id = ? ORDER BY position",
            (match_id,),
        ).fetchall()
    return _row_to_match(m, players)


def delete_match(match_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))
