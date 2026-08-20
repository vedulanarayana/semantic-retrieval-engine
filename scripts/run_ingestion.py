#!/usr/bin/env python
"""Loads a JSONL corpus and ingests it into the FAISS + metadata store.

Each line of the input file should be a JSON object with at least
"id" and "text" fields (a "title" field is used if present):

    {"id": "doc-1", "title": "...", "text": "..."}

Usage:
    python scripts/run_ingestion.py --corpus data/corpus.jsonl --index-type HNSW
"""

import argparse
import json

from app.indexing.faiss_index import FAISSIndex
from app.ingestion.pipeline import IngestionPipeline
from app.storage.metadata_store import MetadataStore


def load_corpus(path: str):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="path to a JSONL file of documents")
    parser.add_argument("--index-type", default="HNSW", choices=["HNSW", "IVF", "Flat"])
    parser.add_argument("--multiprocess", action="store_true", help="use the multiprocessing ingestion path")
    parser.add_argument("--save-index", default=None, help="path to write the built FAISS index to")
    args = parser.parse_args()

    documents = load_corpus(args.corpus)
    print(f"loaded {len(documents)} documents from {args.corpus}")

    pipeline = IngestionPipeline()
    if args.multiprocess:
        embeddings, chunks, elapsed = pipeline.ingest_multiprocess(documents)
    else:
        embeddings, chunks, elapsed = pipeline.ingest_single_process(documents)

    print(f"produced {len(chunks)} chunks in {elapsed:.2f}s ({len(documents) / elapsed:.1f} docs/sec)")

    if len(chunks) == 0:
        print("no chunks produced, nothing to index")
        return

    index = FAISSIndex(args.index_type)
    index.add(embeddings)

    store = MetadataStore()
    store.add_batch(chunks)

    print(f"indexed {index.size()} vectors into a {args.index_type} index")

    if args.save_index:
        index.save(args.save_index)
        print(f"saved index to {args.save_index}")


if __name__ == "__main__":
    main()
