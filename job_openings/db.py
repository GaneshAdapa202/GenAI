"""
db.py
SQLite helper for job postings.
Stores: id, title, company, location, description, url, date_posted, embedding (JSON text)
"""

import sqlite3
import json
from typing import List, Dict, Optional

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        company TEXT,
        location TEXT,
        description TEXT,
        url TEXT UNIQUE,
        date_posted TEXT,
        embedding TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_job(db_path: str, job: Dict):
    """
    job: { title, company, location, description, url, date_posted, embedding (list floats) }
    embedding will be stored as JSON text.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO jobs (title, company, location, description, url, date_posted, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            job.get("title"),
            job.get("company"),
            job.get("location"),
            job.get("description"),
            job.get("url"),
            job.get("date_posted"),
            json.dumps(job.get("embedding")) if job.get("embedding") is not None else None
        ))
        conn.commit()
        inserted = True
    except sqlite3.IntegrityError:
        # URL already exists (unique constraint)
        inserted = False
    finally:
        conn.close()
    return inserted

def list_jobs(db_path: str, limit: int = 50):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, title, company, location, url, date_posted FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_jobs_with_embeddings(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id, title, company, location, description, url, date_posted, embedding FROM jobs")
    rows = []
    for r in c.fetchall():
        emb = json.loads(r[7]) if r[7] else None
        rows.append({
            "id": r[0],
            "title": r[1],
            "company": r[2],
            "location": r[3],
            "description": r[4],
            "url": r[5],
            "date_posted": r[6],
            "embedding": emb
        })
    conn.close()
    return rows
