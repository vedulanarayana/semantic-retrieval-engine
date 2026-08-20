#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

// Brute-force k-nearest-neighbor search: for every query, walk the full
// stored set and keep a bounded top-k. This is the honest O(n) baseline
// the FAISS approximate indexes get compared against — no tricks, so
// its recall is always 1.0 by construction.
class BruteForceKNN {
public:
    enum class Metric { Cosine, L2 };

    BruteForceKNN(std::size_t dim, Metric metric)
        : dim_(dim), metric_(metric) {}

    void add(const float* data, std::size_t num_vectors) {
        vectors_.reserve(vectors_.size() + num_vectors * dim_);
        vectors_.insert(vectors_.end(), data, data + num_vectors * dim_);
    }

    std::size_t size() const {
        return vectors_.size() / dim_;
    }

    // returns (score, index) pairs, sorted best-first — for cosine, score
    // is similarity (higher is better); for L2, score is squared distance
    // (lower is better)
    std::vector<std::pair<float, int>> search(const float* query, std::size_t k) const {
        const std::size_t n = size();
        if (k == 0 || n == 0) {
            return {};
        }
        k = std::min(k, n);

        return metric_ == Metric::Cosine ? topKCosine(query, k, n) : topKL2(query, k, n);
    }

private:
    // keeps a min-heap of the k best (highest) scores seen so far: the
    // weakest of the current top-k sits at the top, ready to be evicted
    // the moment something better shows up. Bounded to O(k) memory even
    // when n is huge.
    std::vector<std::pair<float, int>> topKCosine(const float* query, std::size_t k, std::size_t n) const {
        auto minHeap = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first > b.first;
        };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(minHeap)> heap(minHeap);

        for (std::size_t i = 0; i < n; ++i) {
            float score = cosineSimilarity(query, vectors_.data() + i * dim_);
            if (heap.size() < k) {
                heap.emplace(score, static_cast<int>(i));
            } else if (score > heap.top().first) {
                heap.pop();
                heap.emplace(score, static_cast<int>(i));
            }
        }
        return drain(heap);
    }

    // mirror image of topKCosine: max-heap of the k best (lowest)
    // distances, worst-of-the-best on top so it's cheap to evict
    std::vector<std::pair<float, int>> topKL2(const float* query, std::size_t k, std::size_t n) const {
        auto maxHeap = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
            return a.first < b.first;
        };
        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(maxHeap)> heap(maxHeap);

        for (std::size_t i = 0; i < n; ++i) {
            float score = squaredL2(query, vectors_.data() + i * dim_);
            if (heap.size() < k) {
                heap.emplace(score, static_cast<int>(i));
            } else if (score < heap.top().first) {
                heap.pop();
                heap.emplace(score, static_cast<int>(i));
            }
        }
        return drain(heap);
    }

    template <typename Heap>
    std::vector<std::pair<float, int>> drain(Heap& heap) const {
        std::vector<std::pair<float, int>> results;
        results.reserve(heap.size());
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
        // both heaps pop worst-of-the-remaining-set first, so reversing
        // once always yields best-first order
        std::reverse(results.begin(), results.end());
        return results;
    }

    float cosineSimilarity(const float* a, const float* b) const {
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (std::size_t i = 0; i < dim_; ++i) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0f || normB == 0.0f) {
            return 0.0f;
        }
        return dot / (std::sqrt(normA) * std::sqrt(normB));
    }

    float squaredL2(const float* a, const float* b) const {
        float sum = 0.0f;
        for (std::size_t i = 0; i < dim_; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    std::size_t dim_;
    Metric metric_;
    std::vector<float> vectors_;
};
