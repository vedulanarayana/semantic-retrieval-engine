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


def test_hnsw_uses_inner_product_not_l2():
    # IndexHNSWFlat defaults to METRIC_L2 unless told otherwise — if the
    # HNSW index is ever built without explicitly requesting
    # METRIC_INNER_PRODUCT, its scores silently become L2 distances while
    # every other index type (and the API layer) treats them as cosine
    # similarity, so this must always agree with Flat on identical data
    dim = EMBEDDING_DIM
    vectors = _random_normalized(200, dim)
    query = _random_normalized(1, dim)[0]

    hnsw = FAISSIndex("HNSW")
    hnsw.add(vectors)
    hnsw_scores, hnsw_indices = hnsw.search(query, k=5)

    faiss_flat = FAISSIndex("Flat")
    faiss_flat.add(vectors)
    flat_scores, flat_indices = faiss_flat.search(query, k=5)

    assert set(hnsw_indices.tolist()) == set(flat_indices.tolist())
    # inner-product scores on unit vectors are cosine similarities, so they
    # live in [-1, 1] — squared L2 distances on the same data would not
    assert hnsw_scores.max() <= 1.0 + 1e-4
    assert np.allclose(sorted(hnsw_scores), sorted(flat_scores), atol=1e-4)
