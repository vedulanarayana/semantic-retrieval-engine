import os

# embedding model — small and fast enough to run on CPU, which matters
# since this whole project is designed to run without a GPU
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 384))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 64))

# chunking
CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE_WORDS", 200))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", 40))

# ingestion
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", os.cpu_count() or 4))

# FAISS HNSW
HNSW_M = int(os.environ.get("HNSW_M", 32))
HNSW_EF_CONSTRUCTION = int(os.environ.get("HNSW_EF_CONSTRUCTION", 200))
HNSW_EF_SEARCH = int(os.environ.get("HNSW_EF_SEARCH", 64))

# FAISS IVF
IVF_NLIST = int(os.environ.get("IVF_NLIST", 100))
IVF_NPROBE = int(os.environ.get("IVF_NPROBE", 10))

# metadata store
METADATA_DB_PATH = os.environ.get("METADATA_DB_PATH", "data/metadata.db")

# API
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", 60))

# maps API key -> role. in a real deployment these come from a secrets
# manager, not the environment, but this is enough to make RBAC real
# instead of decorative
API_KEYS = {
    key: role
    for key, role in (
        pair.split(":", 1)
        for pair in os.environ.get("API_KEYS", "dev-admin-key:admin,dev-reader-key:reader").split(",")
        if pair
    )
}
