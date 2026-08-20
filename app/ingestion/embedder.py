from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_BATCH_SIZE


class EmbeddingGenerator:

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        # loaded lazily so a process that only imports this class (e.g. a
        # worker that gets forked before ever calling embed_batch) doesn't
        # pay the model load cost for nothing
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        # normalize_embeddings=True is load-bearing: the FAISS index uses
        # inner product as a stand-in for cosine similarity, and that's
        # only correct if vectors are unit-length going in
        return self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
