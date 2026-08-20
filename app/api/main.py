import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from app.api.auth import require_permission
from app.api.rate_limit import enforce_rate_limit
from app.config import EMBEDDING_DIM
from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingGenerator
from app.indexing.faiss_index import FAISSIndex
from app.storage.metadata_store import MetadataStore

try:
    import knn_cpp
    _brute_force_available = True
except ImportError:
    # the C++ extension is an opt-in build step (scripts/build_cpp.sh) —
    # the API still runs without it, just with the "brute" index disabled
    _brute_force_available = False

app = FastAPI(title="Semantic Retrieval Engine")

chunker = TextChunker()
embedder = EmbeddingGenerator()
metadata_store = MetadataStore()
faiss_index = FAISSIndex("HNSW")
brute_force_index = knn_cpp.BruteForceKNN(EMBEDDING_DIM, "cosine") if _brute_force_available else None


class IngestRequest(BaseModel):
    documents: List[Dict]


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int


class SearchResult(BaseModel):
    doc_id: Optional[str]
    title: Optional[str]
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float
    index_used: str


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(enforce_rate_limit)])
def ingest(request: IngestRequest, _: str = Depends(require_permission("ingest"))):
    embeddings, chunks = [], []
    for doc in request.documents:
        doc_chunks = chunker.chunk_document(doc)
        texts = [c["text"] for c in doc_chunks]
        if not texts:
            continue
        vecs = embedder.embed_batch(texts)
        embeddings.extend(vecs)
        chunks.extend(doc_chunks)

    if not chunks:
        return IngestResponse(chunks_added=0, total_chunks=metadata_store.count())

    embeddings = np.array(embeddings, dtype=np.float32)

    faiss_index.add(embeddings)
    if brute_force_index is not None:
        brute_force_index.add(embeddings)
    metadata_store.add_batch(chunks)

    return IngestResponse(chunks_added=len(chunks), total_chunks=metadata_store.count())


@app.get("/search", response_model=SearchResponse, dependencies=[Depends(enforce_rate_limit)])
def search(q: str, k: int = 10, index: str = "faiss", _: str = Depends(require_permission("search"))):
    if index == "brute" and brute_force_index is None:
        raise HTTPException(status_code=400, detail="brute-force index not built — run scripts/build_cpp.sh")
    if index not in ("faiss", "brute"):
        raise HTTPException(status_code=400, detail="index must be 'faiss' or 'brute'")

    query_vec = embedder.embed_batch([q])[0]

    start = time.time()
    if index == "faiss":
        scores, positions = faiss_index.search(query_vec, k)
    else:
        scores, positions = brute_force_index.search(query_vec, k)
    latency_ms = (time.time() - start) * 1000

    results = []
    for score, position in zip(scores, positions):
        if position < 0:
            continue
        chunk = metadata_store.get(int(position))
        if chunk is None:
            continue
        results.append(SearchResult(doc_id=chunk.get("doc_id"), title=chunk.get("title"), text=chunk["text"], score=float(score)))

    return SearchResponse(results=results, latency_ms=latency_ms, index_used=index)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": metadata_store.count(),
        "brute_force_available": _brute_force_available,
    }
