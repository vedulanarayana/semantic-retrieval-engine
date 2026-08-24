import faiss
import numpy as np

from app.config import EMBEDDING_DIM, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH, HNSW_M, IVF_NLIST, IVF_NPROBE


class FAISSIndex:
    # uses inner product as the distance metric, which only behaves like
    # cosine similarity if every vector fed in is already L2-normalized —
    # the embedder does this, but if that ever changes, this index needs
    # to change with it

    def __init__(self, index_type: str = "HNSW"):
        self.index_type = index_type
        self.dim = EMBEDDING_DIM
        self.index = self._build_index()

    def _build_index(self):
        if self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self.dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = HNSW_EF_SEARCH
            return index
        if self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = IVF_NPROBE
            return index
        return faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray):
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("expected L2-normalized vectors for inner-product search")
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query: np.ndarray, k: int = 10):
        query = query.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query, k)
        return distances[0], indices[0]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def size(self) -> int:
        return self.index.ntotal
