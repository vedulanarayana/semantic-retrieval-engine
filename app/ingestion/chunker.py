from typing import Dict, List

from app.config import CHUNK_OVERLAP_WORDS, CHUNK_SIZE_WORDS


class TextChunker:

    def __init__(self, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size, otherwise chunks never advance")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: Dict) -> List[Dict]:
        text = (doc.get("text") or "").strip()
        if not text:
            return []

        words = text.split()
        step = self.chunk_size - self.overlap
        chunks = []
        for i, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(
                {
                    "doc_id": doc.get("id"),
                    "chunk_index": i,
                    "text": " ".join(window),
                    "title": doc.get("title"),
                }
            )
            if start + self.chunk_size >= len(words):
                break

        return chunks
