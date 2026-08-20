import time
import multiprocessing as mp
from typing import List, Dict, Tuple
import numpy as np

from app.ingestion.chunker import TextChunker
from app.ingestion.embedder import EmbeddingGenerator
from app.config import NUM_WORKERS


def _embed_batch(doc_batch: List[Dict]) -> Tuple[List, List]:
    # runs in a separate process, so it needs its own model instance —
    # can't share the parent's loaded model across a process boundary
    chunker = TextChunker()
    embedder = EmbeddingGenerator()

    embeddings, metadata = [], []
    for doc in doc_batch:
        chunks = chunker.chunk_document(doc)
        texts = [c["text"] for c in chunks]
        if not texts:
            continue
        vecs = embedder.embed_batch(texts)
        embeddings.extend(vecs)
        metadata.extend(chunks)
    return embeddings, metadata


class IngestionPipeline:

    def __init__(self):
        self.chunker = TextChunker()
        self.embedder = EmbeddingGenerator()

    def ingest_single_process(self, documents: List[Dict]) -> Tuple[np.ndarray, List, float]:
        start = time.time()
        embeddings, metadata = [], []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            texts = [c["text"] for c in chunks]
            if texts:
                vecs = self.embedder.embed_batch(texts)
                embeddings.extend(vecs)
                metadata.extend(chunks)
        return np.array(embeddings), metadata, time.time() - start

    def ingest_multiprocess(self, documents: List[Dict], workers: int = NUM_WORKERS) -> Tuple[np.ndarray, List, float]:
        start = time.time()
        batch_size = max(1, len(documents) // workers)
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        with mp.Pool(processes=workers) as pool:
            results = pool.map(_embed_batch, batches)

        embeddings, metadata = [], []
        for emb, meta in results:
            embeddings.extend(emb)
            metadata.extend(meta)

        return np.array(embeddings), metadata, time.time() - start

    def benchmark_throughput(self, documents: List[Dict], sample_size: int = 1000) -> dict:
        sample = documents[:sample_size]
        _, _, single_time = self.ingest_single_process(sample)
        _, _, multi_time = self.ingest_multiprocess(sample)

        return {
            "single_process_seconds": single_time,
            "multiprocess_seconds": multi_time,
            "speedup": single_time / multi_time,
            "docs_per_sec_single": len(sample) / single_time,
            "docs_per_sec_multi": len(sample) / multi_time,
        }
