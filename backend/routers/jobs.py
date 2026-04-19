"""
Store compartilhado de jobs com persistência em SQLite.
Mantém o estado entre reinícios simples do backend.
"""
from pathlib import Path
import sqlite3
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "jobs.sqlite3"


def _connect() -> sqlite3.Connection:
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	conn = sqlite3.connect(DB_PATH)
	conn.execute("PRAGMA journal_mode=WAL")
	conn.row_factory = sqlite3.Row
	return conn


def init_job_store() -> None:
	with _connect() as conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS jobs (
				job_id TEXT PRIMARY KEY,
				status TEXT NOT NULL,
				progress INTEGER NOT NULL DEFAULT 0,
				error TEXT,
				filename TEXT,
				mesh_bytes BLOB,
				created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
				updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
			)
			"""
		)


def create_job(job_id: str, status: str = "processing", progress: int = 0, error: str | None = None) -> None:
	with _connect() as conn:
		conn.execute(
			"""
			INSERT OR REPLACE INTO jobs (job_id, status, progress, error, updated_at)
			VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
			""",
			(job_id, status, progress, error),
		)


def update_job(job_id: str, **fields: Any) -> None:
	if not fields:
		return

	assignments = ", ".join(f"{key} = ?" for key in fields)
	values = list(fields.values())
	values.append(job_id)

	with _connect() as conn:
		conn.execute(
			f"""
			UPDATE jobs
			SET {assignments}, updated_at = CURRENT_TIMESTAMP
			WHERE job_id = ?
			""",
			values,
		)


def get_job(job_id: str) -> dict[str, Any] | None:
	with _connect() as conn:
		row = conn.execute(
			"SELECT job_id, status, progress, error, filename, mesh_bytes, created_at, updated_at FROM jobs WHERE job_id = ?",
			(job_id,),
		).fetchone()

	return dict(row) if row else None


init_job_store()
