from app.ingestion.chunker import TextChunker


def test_chunker_splits_long_text_with_overlap():
    chunker = TextChunker(chunk_size=10, overlap=2)
    doc = {"id": "doc-1", "title": "test", "text": " ".join(f"word{i}" for i in range(25))}

    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    assert all(c["doc_id"] == "doc-1" for c in chunks)
    # consecutive chunks should share the overlap words
    first_tail = chunks[0]["text"].split()[-2:]
    second_head = chunks[1]["text"].split()[:2]
    assert first_tail == second_head


def test_chunker_handles_empty_text():
    chunker = TextChunker()
    assert chunker.chunk_document({"id": "doc-1", "text": ""}) == []


def test_chunker_rejects_overlap_not_smaller_than_chunk_size():
    import pytest
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=10)
