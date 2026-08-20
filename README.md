# semantic-retrieval-engine

A semantic search / document retrieval service: chunk documents, embed them
locally with `sentence-transformers`, index the vectors two different ways,
and query both through a FastAPI endpoint. The point of building two index
paths is to have a real comparison instead of a single unverified number —
a brute-force C++ k-NN search gives exact recall by construction, and FAISS
(HNSW or IVF) gives the approximate, production-shaped alternative you'd
actually deploy.

## Architecture

```
Ingestion
  documents -> chunker -> embedder (sentence-transformers, normalized) -> vectors

Indexing (two paths, so there's something to compare)
  FAISS (HNSW / IVF / Flat)     <- approximate, what you'd run in production
  C++ brute-force k-NN (pybind11) <- exact, used as ground truth for recall

Query layer (FastAPI)
  embed query -> search chosen index -> map positions back to chunk text
  RBAC (API key -> role) + a sliding-window rate limiter sit in front of it

Evaluation
  benchmarks/benchmark_recall.py runs the same query set against every
  index type and reports recall@k and latency against the brute-force
  ground truth
```

## Why multiprocessing, not multithreading, for ingestion

Embedding text is CPU-bound work, and Python threads don't get real
parallelism on CPU-bound work because of the GIL. `IngestionPipeline`
uses `multiprocessing.Pool` for the parallel path instead, with each
worker process loading its own model instance (you can't share a loaded
model handle across a process boundary). `benchmark_throughput()` measures
single-process vs. multiprocess wall-clock time directly — it doesn't
assume a speedup, it times both and reports the ratio it actually got.

## The inner-product / cosine dependency

`FAISSIndex` uses inner product (`IndexFlatIP`, `METRIC_INNER_PRODUCT`) as
its distance metric. Inner product only equals cosine similarity if every
vector going in is L2-normalized first. `EmbeddingGenerator.embed_batch`
sets `normalize_embeddings=True` for exactly this reason, and `add()`
checks vector norms and raises if anything unnormalized slips through,
so a future change to the embedding call fails loudly instead of quietly
returning wrong neighbors.

The C++ `BruteForceKNN` doesn't have this dependency — its cosine mode
divides by the actual vector norms, so it's correct regardless of whether
the input happens to be normalized.

## Running it

```bash
pip install -r requirements.txt

# build the C++ extension (needed for the brute-force index and its tests)
./scripts/build_cpp.sh

# ingest a corpus (JSONL, one {"id", "title", "text"} object per line)
python scripts/run_ingestion.py --corpus data/corpus.jsonl --index-type HNSW

# run the API
uvicorn app.api.main:app --reload
```

Default dev API keys (override with the `API_KEYS` env var,
`key:role,key:role` format): `dev-admin-key` (admin), `dev-reader-key`
(reader). Pass one as the `X-API-Key` header.

```
POST /ingest   (admin)   {"documents": [{"id": "...", "title": "...", "text": "..."}]}
GET  /search   (reader)  ?q=...&k=10&index=faiss|brute
GET  /health
```

`/search` returns which index actually served the request and the
measured latency, so the two paths stay directly comparable from the
outside, not just in the offline benchmark.

## Benchmarking recall

```python
from benchmarks.benchmark_recall import run_benchmark
results = run_benchmark(embeddings, queries, k=10)
```

Ground truth is computed per query (a `Flat` index search for that
specific query's embedding), not shared across queries — an earlier draft
reused a single ground-truth result for every query in the set, which
would have made the whole recall number meaningless. This runs a full
brute-force pass plus three index builds every time; fine for a one-off
comparison, worth restructuring into a single shared indexing pass if it
ever needs to run often.

## Tests

```bash
pytest tests/
```

`test_knn.py` and `test_faiss_index.py` need the compiled extension and
skip themselves with a clear reason if it isn't built yet. `test_faiss_index.py`
in particular checks that FAISS's exact (`Flat`) index and the C++
brute-force index agree on identical data — if they ever don't, that's a
real bug in one of the two, not a benchmarking artifact.

## What's still a placeholder

Corpus size, recall@k, multiprocessing speedup, and P95 latency are real
numbers only once you've actually run ingestion and the benchmark against
a real corpus and (if you deploy it) a real Cloud Run instance. Local and
deployed latency are genuinely different numbers worth keeping separately
rather than picking one to report.
