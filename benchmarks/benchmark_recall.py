import time
import numpy as np
from typing import List

from app.indexing.faiss_index import FAISSIndex


def compute_ground_truth(embeddings: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    # exact search, used only to know what the "correct" answer is —
    # this becomes the baseline every approximate index gets compared against
    flat = FAISSIndex("Flat")
    flat.add(embeddings)
    return np.array([flat.search(q, k)[1] for q in queries])


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    return len(set(retrieved) & set(ground_truth)) / k


def run_benchmark(embeddings: np.ndarray, queries: np.ndarray, k: int = 10) -> List[dict]:
    ground_truth = compute_ground_truth(embeddings, queries, k)
    results = []

    for index_type in ["HNSW", "IVF", "Flat"]:
        index = FAISSIndex(index_type)
        index.add(embeddings)

        latencies, recalls = [], []
        for i, query in enumerate(queries):
            start = time.time()
            _, indices = index.search(query, k)
            latencies.append((time.time() - start) * 1000)
            recalls.append(recall_at_k(indices, ground_truth[i], k))

        results.append({
            "index_type": index_type,
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "recall_at_10": np.mean(recalls),
        })

    return results
