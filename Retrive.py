"""
embed_and_retrieve.py — Milestone 4: Embedding + Vector Store + Retrieval
UCF CS Unofficial Guide RAG Pipeline

Architecture (from planning.md):
  chunks.json  →  all-MiniLM-L6-v2  →  ChromaDB  →  retrieve(query, k=5)

What each section does
──────────────────────
1. load_chunks()
   Reads chunks.json produced by pipeline.py.
   Returns a list of dicts with keys:
     chunk_id, source_id, source_name, url, text, char_count

2. build_vectorstore()
   - Loads the SentenceTransformer model locally (no API key needed).
   - Calls model.encode() to turn every chunk's text into a 384-dim float vector.
   - Creates (or reloads) a persistent ChromaDB collection on disk at ./chroma_db/
   - Calls collection.add() to store:
       documents  — the raw chunk text  (ChromaDB stores this for you)
       embeddings — the float vectors   (used for similarity search)
       metadatas  — source_name, url, source_id, chunk_position (for attribution)
       ids        — chunk_id strings    (must be unique; used to avoid re-adding)

3. retrieve(query, k=5)
   - Embeds the query string with the same model.
   - Calls collection.query() which computes cosine similarity between the
     query vector and every stored vector, then returns the k closest.
   - Returns a list of dicts: text + all metadata fields.

ChromaDB API notes (so you know what each call does)
─────────────────────────────────────────────────────
  chromadb.PersistentClient(path)
      Opens (or creates) a local SQLite-backed vector store at `path`.
      Data survives between Python runs — you only embed once.

  client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
      Gets the named collection if it exists, else creates it.
      hnsw:space="cosine" tells ChromaDB to use cosine similarity
      instead of the default L2 (Euclidean) distance.
      Cosine similarity is better for text because it measures
      direction (meaning) not magnitude (length).

  collection.add(ids, embeddings, documents, metadatas)
      Inserts records. ids must be unique strings — if you call add()
      with an id that already exists ChromaDB raises an error, which is
      why we check existing ids first and skip duplicates.

  collection.query(query_embeddings, n_results, include)
      Returns the n_results nearest neighbours to the query vector.
      include controls which fields come back:
        "documents"  — the stored text
        "metadatas"  — the dict you passed into add()
        "distances"  — cosine distance (0 = identical, 2 = opposite)

Usage:
    pip install sentence-transformers chromadb
    python embed_and_retrieve.py                        # embed + run test queries
    python embed_and_retrieve.py --rebuild              # wipe DB and re-embed
    python embed_and_retrieve.py --query "your question here"
"""

import argparse
import json
import os
import shutil
import sys
import time

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_FILE    = "chunks.json"
CHROMA_PATH    = "./chroma_db"
COLLECTION_NAME = "ucf_cs_guide"
EMBED_MODEL    = "all-MiniLM-L6-v2"   # 384-dim, runs locally, no API key
TOP_K          = 5                     # from planning.md
BATCH_SIZE     = 64                    # how many chunks to embed at once

# ── Stage 1: Load chunks ──────────────────────────────────────────────────────

def load_chunks(path: str = CHUNKS_FILE) -> list[dict]:
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] {path} not found.\n"
            "Run pipeline.py first to generate chunks.json."
        )
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks):,} chunks from {path}")
    return chunks


# ── Stage 2: Embed + store in ChromaDB ───────────────────────────────────────

def build_vectorstore(
    chunks: list[dict],
    rebuild: bool = False,
) -> tuple[chromadb.Collection, SentenceTransformer]:
    """
    Embed all chunks and store them in ChromaDB.

    Parameters
    ----------
    chunks  : list of chunk dicts from pipeline.py
    rebuild : if True, wipe the existing DB and start fresh

    Returns
    -------
    collection : ChromaDB collection (ready to query)
    model      : loaded SentenceTransformer (reused in retrieve())
    """

    # Optionally wipe existing DB
    if rebuild and os.path.exists(CHROMA_PATH):
        print(f"  [rebuild] Removing existing DB at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    # ── Load embedding model ─────────────────────────────────────────────────
    # SentenceTransformer downloads the model on first run (~90 MB),
    # then caches it locally so subsequent runs are instant.
    print(f"\n  Loading embedding model: {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  Model loaded  (embedding dim: {model.get_sentence_embedding_dimension()})")

    # ── Open / create ChromaDB ───────────────────────────────────────────────
    # PersistentClient writes to disk so embeddings survive between runs.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # get_or_create_collection: safe to call even if collection already exists.
    # hnsw:space="cosine" → use cosine similarity (better for text than L2).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Skip chunks already in the DB ────────────────────────────────────────
    # collection.get() returns all stored ids so we can diff.
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks   = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print(f"\n  Collection already contains {len(existing_ids):,} chunks — nothing to add.")
        print("  (Use --rebuild to re-embed from scratch.)")
        return collection, model

    print(f"\n  Embedding {len(new_chunks):,} new chunks in batches of {BATCH_SIZE} …")
    t0 = time.time()

    # Process in batches to avoid memory spikes on large corpora
    for batch_start in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[batch_start : batch_start + BATCH_SIZE]

        texts      = [c["text"]        for c in batch]
        ids        = [c["chunk_id"]    for c in batch]

        # metadatas is a list of dicts — one per chunk.
        # These are stored alongside the vector and returned on every query,
        # so include everything you'll want to show the user.
        metadatas  = [
            {
                "source_name":     c["source_name"],
                "url":             c["url"],
                "source_id":       c["source_id"],
                "chunk_position":  int(c["chunk_id"].split("chunk")[-1]),  # e.g. 42
            }
            for c in batch
        ]

        # model.encode() returns a numpy array of shape (batch_size, 384).
        # .tolist() converts it to a plain Python list that ChromaDB accepts.
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # collection.add() stores vectors + text + metadata together.
        # The documents= field lets you get the text back in query results
        # without keeping a separate lookup table.
        collection.add(
            ids        = ids,
            embeddings = embeddings,
            documents  = texts,
            metadatas  = metadatas,
        )

        done = min(batch_start + BATCH_SIZE, len(new_chunks))
        print(f"    [{done:>5}/{len(new_chunks)}] embedded", end="\r")

    elapsed = time.time() - t0
    total   = collection.count()
    print(f"\n  ✓ Done in {elapsed:.1f}s  |  collection total: {total:,} chunks")

    return collection, model


# ── Stage 3: Retrieval ────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    k: int = TOP_K,
) -> list[dict]:
    """
    Return the top-k most relevant chunks for a query string.

    How it works
    ─────────────
    1. Embed the query with the same model used during indexing.
       Using the same model is critical — different models produce
       incompatible vector spaces.
    2. collection.query() computes cosine distance between the query
       vector and every stored vector using the HNSW index.
       Distance 0 = identical meaning, 2 = opposite meaning.
    3. Results come back sorted by distance (closest first).
    4. We package each result as a dict with text + metadata for easy use
       in the generation stage.

    Parameters
    ----------
    query      : natural-language question
    collection : loaded ChromaDB collection
    model      : the same SentenceTransformer used to build the store
    k          : number of chunks to return (default 5, from planning.md)

    Returns
    -------
    List of dicts, each with:
      text, source_name, url, source_id, chunk_position, distance
    """
    query_embedding = model.encode([query]).tolist()

    # collection.query() returns a dict of parallel lists:
    #   results["documents"][0]  — list of k texts
    #   results["metadatas"][0]  — list of k metadata dicts
    #   results["distances"][0]  — list of k cosine distances
    # The [0] index is because query() supports batching; we send one query.
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":           text,
            "source_name":    meta["source_name"],
            "url":            meta["url"],
            "source_id":      meta["source_id"],
            "chunk_position": meta["chunk_position"],
            "distance":       round(dist, 4),   # lower = more relevant
        })

    return output


# ── Pretty-print helper ───────────────────────────────────────────────────────

def print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'═'*60}")
    print(f"Query: {query}")
    print(f"Top {len(results)} results")
    print("═" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['source_name']}  (distance: {r['distance']})")
        print(f"    {r['url']}")
        print(f"    chunk position: {r['chunk_position']}")
        print(f"    ── text ──────────────────────────────────────────")
        # Wrap at 72 chars for readability
        words, line = r["text"].split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 72:
                print(f"    {line}")
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            print(f"    {line}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    "What clubs should a student interested in software engineering join?",
    "Where can UCF students find internship opportunities?",
    "Which UCF organization focuses on artificial intelligence?",
    "What resources help students prepare for careers in tech?",
    "Which club is best known for hackathons?",
]


def main():
    parser = argparse.ArgumentParser(description="Embed and retrieve — UCF CS RAG")
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe ChromaDB and re-embed all chunks")
    parser.add_argument("--query", type=str, default=None,
                        help="Run a single retrieval query and exit")
    parser.add_argument("--k", type=int, default=TOP_K,
                        help=f"Number of chunks to retrieve (default {TOP_K})")
    args = parser.parse_args()

    print("═" * 60)
    print("Milestone 4 — Embedding + Retrieval")
    print(f"  Model : {EMBED_MODEL}")
    print(f"  Top-k : {args.k}")
    print(f"  DB    : {CHROMA_PATH}")
    print("═" * 60)

    # Load chunks from pipeline output
    chunks = load_chunks()

    # Build (or reload) the vector store
    collection, model = build_vectorstore(chunks, rebuild=args.rebuild)

    # Run a single query if provided
    if args.query:
        results = retrieve(args.query, collection, model, k=args.k)
        print_results(args.query, results)
        return

    # Otherwise run all 5 evaluation queries from planning.md
    print("\n── Running evaluation queries from planning.md ───────────────")
    for q in TEST_QUERIES:
        results = retrieve(q, collection, model, k=args.k)
        print_results(q, results)


if __name__ == "__main__":
    main()