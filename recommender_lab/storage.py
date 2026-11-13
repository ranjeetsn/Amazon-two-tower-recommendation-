import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path("recommender_lab/data/interactions.sqlite3")


def database_path():
    path = Path(os.getenv("RECOMMENDER_DB_PATH", str(DEFAULT_DB)))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect():
    conn = sqlite3.connect(database_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS searches (
                request_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ranking',
                error TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS recommendations (
                request_id TEXT NOT NULL REFERENCES searches(request_id),
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                product_id TEXT NOT NULL,
                product_title TEXT NOT NULL,
                rank INTEGER NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (request_id, model_name, product_id)
            );

            CREATE TABLE IF NOT EXISTS feedback (
                request_id TEXT NOT NULL REFERENCES searches(request_id),
                product_id TEXT NOT NULL,
                product_title TEXT NOT NULL,
                label TEXT NOT NULL CHECK (label IN ('E', 'S', 'C', 'I')),
                rationale TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (request_id, product_id, source)
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL,
                feedback_count INTEGER NOT NULL DEFAULT 0,
                model_version TEXT,
                message TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_recommendations_model
                ON recommendations(model_name);
            CREATE INDEX IF NOT EXISTS idx_feedback_created
                ON feedback(created_at);
            """
        )

        # Migrate databases created by earlier versions of the local lab.
        search_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(searches)")
        }
        if "status" not in search_columns:
            # Older rows were only written after ranking completed.
            conn.execute(
                "ALTER TABLE searches ADD COLUMN status TEXT NOT NULL DEFAULT 'complete'"
            )
        if "error" not in search_columns:
            conn.execute(
                "ALTER TABLE searches ADD COLUMN error TEXT NOT NULL DEFAULT ''"
            )

        conn.execute(
            "INSERT OR IGNORE INTO app_state(key, value) "
            "VALUES('last_trained_feedback_count', '0')"
        )


def record_query(query):
    """Persist the incoming query before any recommendation model is called."""
    query = str(query).strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    init_db()
    request_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO searches(request_id, query, created_at, status)
            VALUES (?, ?, ?, 'ranking')
            """,
            (request_id, query, utc_now()),
        )
    return request_id


def record_recommendations(request_id, results_by_model, model_version="baseline"):
    """Attach all model outputs to a previously logged query."""
    init_db()
    rows = []
    for model_name, results in results_by_model.items():
        for rank, item in enumerate(results, start=1):
            rows.append(
                (
                    request_id,
                    model_name,
                    model_version,
                    str(item["product_id"]),
                    str(item.get("product_title", "")),
                    rank,
                    float(item.get("score", 0.0)),
                )
            )

    with connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM searches WHERE request_id=?", (request_id,)
        ).fetchone()
        if exists is None:
            raise ValueError(f"Unknown request_id: {request_id}")

        conn.executemany(
            """
            INSERT OR REPLACE INTO recommendations
            (request_id, model_name, model_version, product_id, product_title, rank, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.execute(
            "UPDATE searches SET status='complete', error='' WHERE request_id=?",
            (request_id,),
        )


def mark_search_failed(request_id, error):
    """Keep the query event and record why recommendation generation failed."""
    init_db()
    with connect() as conn:
        conn.execute(
            "UPDATE searches SET status='failed', error=? WHERE request_id=?",
            (str(error)[:2000], request_id),
        )


def record_search(query, results_by_model, model_version="baseline"):
    """Backward-compatible convenience wrapper for older callers."""
    request_id = record_query(query)
    try:
        record_recommendations(request_id, results_by_model, model_version)
    except Exception as exc:
        mark_search_failed(request_id, exc)
        raise
    return request_id


def record_feedback(request_id, ratings, source="llm_simulated"):
    init_db()
    created_at = utc_now()
    rows = [
        (
            request_id,
            str(item["product_id"]),
            str(item.get("product_title", "")),
            str(item["label"]).upper(),
            str(item.get("rationale", ""))[:500],
            source,
            created_at,
        )
        for item in ratings
        if str(item.get("label", "")).upper() in {"E", "S", "C", "I"}
    ]

    with connect() as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO feedback
            (request_id, product_id, product_title, label, rationale, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def feedback_count():
    init_db()
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


def last_trained_feedback_count():
    init_db()
    with connect() as conn:
        value = conn.execute(
            "SELECT value FROM app_state WHERE key='last_trained_feedback_count'"
        ).fetchone()[0]
    return int(value)


def new_feedback_count():
    return max(0, feedback_count() - last_trained_feedback_count())


def load_training_feedback():
    init_db()
    with connect() as conn:
        frame = pd.read_sql_query(
            """
            SELECT f.request_id AS query_id, s.query, f.product_id,
                   f.product_title, f.label AS esci_label, f.source
            FROM feedback f
            JOIN searches s ON s.request_id = f.request_id
            ORDER BY (f.source = 'human') DESC, f.created_at ASC
            """,
            conn,
        )
    if frame.empty:
        return frame
    return frame.drop_duplicates(["query_id", "product_id"], keep="first")


def mark_training_started(count):
    init_db()
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO training_runs(started_at, status, feedback_count)
            VALUES (?, 'running', ?)
            """,
            (utc_now(), int(count)),
        )
        return cursor.lastrowid


def mark_training_finished(
    run_id,
    status,
    model_version="",
    message="",
    trained_through_feedback_count=None,
):
    init_db()
    with connect() as conn:
        conn.execute(
            """
            UPDATE training_runs
            SET finished_at=?, status=?, model_version=?, message=?
            WHERE run_id=?
            """,
            (utc_now(), status, model_version, str(message)[:2000], int(run_id)),
        )
        if status == "succeeded":
            trained_through = trained_through_feedback_count
            if trained_through is None:
                trained_through = conn.execute(
                    "SELECT COUNT(*) FROM feedback"
                ).fetchone()[0]
            conn.execute(
                "UPDATE app_state SET value=? WHERE key='last_trained_feedback_count'",
                (str(int(trained_through)),),
            )


def get_dashboard_frames():
    init_db()
    with connect() as conn:
        searches = pd.read_sql_query(
            "SELECT * FROM searches ORDER BY created_at DESC", conn
        )
        recommendations = pd.read_sql_query(
            "SELECT * FROM recommendations ORDER BY request_id, model_name, rank",
            conn,
        )
        feedback = pd.read_sql_query(
            "SELECT * FROM feedback ORDER BY created_at DESC",
            conn,
        )
        training_runs = pd.read_sql_query(
            "SELECT * FROM training_runs ORDER BY run_id DESC",
            conn,
        )
    return searches, recommendations, feedback, training_runs
