import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from app.config import METADATA_DB_PATH


class MetadataStore:
    # maps a vector's position in the FAISS/brute-force index (its
    # insertion order, which is what search() returns as an index) to the
    # chunk it actually came from. positions are assigned sequentially and
    # never reused, so this only works if callers always add to the vector
    # index and this store together, in the same order.

    def __init__(self, db_path: str = METADATA_DB_PATH):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                position INTEGER PRIMARY KEY,
                doc_id TEXT,
                title TEXT,
                text TEXT,
                extra TEXT
            )
        """)
        self.conn.commit()

    def next_position(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM chunks").fetchone()
        return row[0]

    def add_batch(self, chunks: List[Dict]) -> List[int]:
        start = self.next_position()
        positions = list(range(start, start + len(chunks)))
        self.conn.executemany(
            "INSERT INTO chunks (position, doc_id, title, text, extra) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    pos,
                    chunk.get("doc_id"),
                    chunk.get("title"),
                    chunk.get("text"),
                    json.dumps({k: v for k, v in chunk.items() if k not in ("doc_id", "title", "text")}),
                )
                for pos, chunk in zip(positions, chunks)
            ],
        )
        self.conn.commit()
        return positions

    def get(self, position: int) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT doc_id, title, text, extra FROM chunks WHERE position = ?", (position,)
        ).fetchone()
        if row is None:
            return None
        doc_id, title, text, extra = row
        return {"doc_id": doc_id, "title": title, "text": text, **json.loads(extra)}

    def get_many(self, positions: List[int]) -> List[Optional[Dict]]:
        return [self.get(p) for p in positions]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
