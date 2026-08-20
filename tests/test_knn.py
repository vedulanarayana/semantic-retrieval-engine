import numpy as np
import pytest

knn_cpp = pytest.importorskip("knn_cpp", reason="C++ extension not built — run scripts/build_cpp.sh first")


def test_l2_finds_the_actually_closest_points():
    # points on a line: 0, 1, 2, 5, 10 — nearest neighbors to 1.5 are
    # obvious by hand, so this catches a wrong heap direction immediately
    points = np.array([[0.0], [1.0], [2.0], [5.0], [10.0]], dtype=np.float32)
    index = knn_cpp.BruteForceKNN(dim=1, metric="l2")
    index.add(points)

    scores, indices = index.search(np.array([1.5], dtype=np.float32), k=2)

    assert set(indices.tolist()) == {1, 2}
    assert scores[0] <= scores[1]


def test_cosine_ranks_by_angle_not_magnitude():
    vectors = np.array(
        [
            [1.0, 0.0],  # same direction as query
            [0.0, 1.0],  # orthogonal
            [100.0, 0.0],  # same direction, huge magnitude — should tie with the first
        ],
        dtype=np.float32,
    )
    index = knn_cpp.BruteForceKNN(dim=2, metric="cosine")
    index.add(vectors)

    scores, indices = index.search(np.array([1.0, 0.0], dtype=np.float32), k=1)

    assert indices[0] in (0, 2)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_dataset_returns_everything():
    vectors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    index = knn_cpp.BruteForceKNN(dim=2, metric="l2")
    index.add(vectors)

    _, indices = index.search(np.array([0.0, 0.0], dtype=np.float32), k=10)

    assert len(indices) == 2
