import tempfile
from pathlib import Path

import pytest

from app.storage.metadata_store import MetadataStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        yield MetadataStore(db_path=str(Path(tmp) / "metadata.db"))


def test_get_many_returns_chunks_in_requested_order(store):
    positions = store.add_batch(
        [
            {"doc_id": "a", "title": "A", "text": "first"},
            {"doc_id": "b", "title": "B", "text": "second"},
            {"doc_id": "c", "title": "C", "text": "third"},
        ]
    )

    # request out of insertion order, on purpose
    results = store.get_many([positions[2], positions[0], positions[1]])

    assert [r["text"] for r in results] == ["third", "first", "second"]


def test_get_many_returns_none_for_missing_positions(store):
    positions = store.add_batch([{"doc_id": "a", "title": "A", "text": "only"}])

    results = store.get_many([positions[0], 999])

    assert results[0]["text"] == "only"
    assert results[1] is None


def test_get_many_empty_input_returns_empty_list(store):
    assert store.get_many([]) == []
