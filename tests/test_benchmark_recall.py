import numpy as np

from benchmarks.benchmark_recall import recall_at_k


def test_recall_at_k_full_match():
    retrieved = np.array([1, 2, 3])
    ground_truth = np.array([3, 2, 1])
    assert recall_at_k(retrieved, ground_truth, k=3) == 1.0


def test_recall_at_k_ignores_padding_from_small_datasets():
    # FAISS pads with -1 when a query asks for more neighbors than exist —
    # those padding entries must not count as a "match" between two
    # independently-padded result sets
    retrieved = np.array([1, -1, -1])
    ground_truth = np.array([1, -1, -1])
    assert recall_at_k(retrieved, ground_truth, k=3) == 1 / 3


def test_recall_at_k_no_overlap():
    retrieved = np.array([1, 2])
    ground_truth = np.array([3, 4])
    assert recall_at_k(retrieved, ground_truth, k=2) == 0.0
