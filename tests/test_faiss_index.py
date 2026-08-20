import numpy as np
import pytest

from app.config import EMBEDDING_DIM
from app.indexing.faiss_index import FAISSIndex

knn_cpp = pytest.importorskip("knn_cpp", reason="C++ extension not built — run scripts/build_cpp.sh first")


def _random_normalized(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_flat_faiss_agrees_with_brute_force_cpp():
    # on identical data, exact search should return the exact same
    # neighbors regardless of which implementation does the searching —
    # if these ever disagree, one of the two has a real bug
    dim = EMBEDDING_DIM
    vectors = _random_normalized(200, dim)
    query = _random_normalized(1, dim)[0]

    faiss_flat = FAISSIndex("Flat")
    faiss_flat.add(vectors)
    _, faiss_indices = faiss_flat.search(query, k=5)

    brute = knn_cpp.BruteForceKNN(dim=dim, metric="cosine")
    brute.add(vectors)
    _, brute_indices = brute.search(query, k=5)

    assert set(faiss_indices.tolist()) == set(brute_indices.tolist())


def test_add_rejects_unnormalized_vectors():
    index = FAISSIndex("Flat")
    unnormalized = np.array([[3.0, 4.0] + [0.0] * (index.dim - 2)], dtype=np.float32)

    with pytest.raises(ValueError):
        index.add(unnormalized)
