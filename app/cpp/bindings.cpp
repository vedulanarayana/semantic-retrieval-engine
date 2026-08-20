#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "knn.h"

namespace py = pybind11;

// thin wrapper so the Python side deals in numpy arrays instead of raw
// pointers — BruteForceKNN itself stays free of any pybind11 dependency
class PyBruteForceKNN {
public:
    PyBruteForceKNN(std::size_t dim, const std::string& metric)
        : impl_(dim, parseMetric(metric)), dim_(dim) {}

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> vectors) {
        auto buf = vectors.request();
        if (buf.ndim != 2 || static_cast<std::size_t>(buf.shape[1]) != dim_) {
            throw std::invalid_argument("expected a 2D array with shape (n, dim)");
        }
        impl_.add(static_cast<const float*>(buf.ptr), buf.shape[0]);
    }

    py::tuple search(py::array_t<float, py::array::c_style | py::array::forcecast> query, std::size_t k) {
        auto buf = query.request();
        if (buf.size != static_cast<py::ssize_t>(dim_)) {
            throw std::invalid_argument("query dimension does not match index dimension");
        }
        auto hits = impl_.search(static_cast<const float*>(buf.ptr), k);

        auto scores = py::array_t<float>(hits.size());
        auto indices = py::array_t<int>(hits.size());
        auto scoresView = scores.mutable_unchecked<1>();
        auto indicesView = indices.mutable_unchecked<1>();
        for (std::size_t i = 0; i < hits.size(); ++i) {
            scoresView(i) = hits[i].first;
            indicesView(i) = hits[i].second;
        }
        return py::make_tuple(scores, indices);
    }

    std::size_t size() const {
        return impl_.size();
    }

private:
    static BruteForceKNN::Metric parseMetric(const std::string& metric) {
        if (metric == "cosine") return BruteForceKNN::Metric::Cosine;
        if (metric == "l2") return BruteForceKNN::Metric::L2;
        throw std::invalid_argument("metric must be 'cosine' or 'l2'");
    }

    BruteForceKNN impl_;
    std::size_t dim_;
};

PYBIND11_MODULE(knn_cpp, m) {
    m.doc() = "Brute-force k-NN, used as the exact-recall baseline for the FAISS benchmarks";

    py::class_<PyBruteForceKNN>(m, "BruteForceKNN")
        .def(py::init<std::size_t, std::string>(), py::arg("dim"), py::arg("metric") = "cosine")
        .def("add", &PyBruteForceKNN::add)
        .def("search", &PyBruteForceKNN::search, py::arg("query"), py::arg("k") = 10)
        .def("size", &PyBruteForceKNN::size);
}
