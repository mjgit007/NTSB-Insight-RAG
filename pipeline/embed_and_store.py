"""
embed_and_store.py - Embed NTSB chunks with Gemini and store in ChromaDB

Stages:
  1. Load chunks from JSONL (output of ingest.py)
  2. Filter out chunks already in ChromaDB (resume support)
  3. Batch 100 chunks -> Gemini text-embedding-004 -> 768-dim vectors
  4. Upsert vectors + text + metadata into ChromaDB

Usage:
  export GOOGLE_API_KEY=your_key_here
  python embed_and_store.py --jsonl chunks_2020_2024.jsonl

Requirements:
  pip install google-genai chromadb
"""

import os
import json
import time
import logging
import argparse

import chromadb
from google import genai
from google.genai import types

# Auto-load .env from AI/RAG/ directory
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBED_MODEL    = 'gemini-embedding-001'  # Gemini free embedding model, 768 dims
BATCH_SIZE     = 100                     # Gemini supports up to 100 texts per batch
RETRY_DELAY    = 5.0                     # seconds to wait on API error before retry
MAX_RETRIES    = 3                       # max retries per batch
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectordb')
COLLECTION     = 'ntsb_reports'

# ---------------------------------------------------------------------------
# Sanitize metadata for ChromaDB
# ChromaDB only accepts: str, int, float, bool — no None values
# ---------------------------------------------------------------------------
def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if v is None:
            clean[k] = ''           # None -> empty string
        elif isinstance(v, float) and v != v:
            clean[k] = ''           # NaN -> empty string
        elif isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)       # fallback: convert to string
    return clean


# ---------------------------------------------------------------------------
# Embed a batch of texts using Gemini text-embedding-004
# Returns list of 768-dim float vectors
# ---------------------------------------------------------------------------
def embed_batch(client: genai.Client, texts: list[str], attempt: int = 1) -> list[list[float]]:
    try:
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT')
        )
        return [e.values for e in response.embeddings]

    except Exception as e:
        if attempt <= MAX_RETRIES:
            wait = RETRY_DELAY * attempt
            log.warning(f"  Embedding error (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
            return embed_batch(client, texts, attempt + 1)
        log.error(f"  Embedding failed after {MAX_RETRIES} attempts: {e}")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Embed NTSB chunks and store in ChromaDB')
    parser.add_argument('--jsonl',      required=True, help='Input JSONL file from ingest.py')
    parser.add_argument('--db-path',    default=CHROMA_DB_PATH, help='ChromaDB persistence directory')
    parser.add_argument('--collection', default=COLLECTION,     help='ChromaDB collection name')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Chunks per embedding API call')
    args = parser.parse_args()

    # API key check
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        log.error("GOOGLE_API_KEY environment variable not set.")
        log.error("Export it with: export GOOGLE_API_KEY=your_key_here")
        return

    # Init Gemini client
    gemini = genai.Client(api_key=api_key)
    log.info(f"Gemini client ready. Model: {EMBED_MODEL}")

    # Init ChromaDB
    os.makedirs(args.db_path, exist_ok=True)
    chroma  = chromadb.PersistentClient(path=args.db_path)
    col     = chroma.get_or_create_collection(
        name=args.collection,
        metadata={'hnsw:space': 'cosine'}   # cosine similarity for semantic search
    )
    log.info(f"ChromaDB ready at: {args.db_path}")
    log.info(f"Collection: '{args.collection}' | Existing chunks: {col.count()}")

    # Load chunks from JSONL
    jsonl_path = args.jsonl if os.path.isabs(args.jsonl) else os.path.join(os.getcwd(), args.jsonl)
    with open(jsonl_path) as f:
        all_chunks = [json.loads(line) for line in f]
    log.info(f"Loaded {len(all_chunks)} chunks from {jsonl_path}")

    # Resume support: find which IDs already exist in ChromaDB
    existing_ids = set()
    if col.count() > 0:
        existing = col.get(include=[])   # fetch only IDs, no embeddings
        existing_ids = set(existing['ids'])
        log.info(f"Skipping {len(existing_ids)} already-stored chunks (resume)")

    # Filter to only new chunks
    new_chunks = [c for c in all_chunks if c['id'] not in existing_ids]
    log.info(f"Chunks to embed and store: {len(new_chunks)}")

    if not new_chunks:
        log.info("Nothing to do — all chunks already stored.")
        return

    # Process in batches
    stats       = {'stored': 0, 'failed': 0}
    total       = len(new_chunks)
    num_batches = (total + args.batch_size - 1) // args.batch_size

    for batch_num in range(num_batches):
        start = batch_num * args.batch_size
        end   = min(start + args.batch_size, total)
        batch = new_chunks[start:end]

        log.info(f"--- Batch {batch_num + 1}/{num_batches} | Chunks {start + 1}-{end} of {total} ---")

        texts     = [c['text'] for c in batch]
        ids       = [c['id']   for c in batch]
        metadatas = [sanitize_metadata(c['metadata']) for c in batch]

        try:
            # Embed
            embeddings = embed_batch(gemini, texts)
            log.info(f"  Embedded {len(embeddings)} chunks (dim={len(embeddings[0])})")

            # Upsert into ChromaDB
            col.upsert(
                ids        = ids,
                embeddings = embeddings,
                documents  = texts,
                metadatas  = metadatas
            )
            log.info(f"  Stored in ChromaDB. Collection total: {col.count()}")
            stats['stored'] += len(batch)

        except Exception as e:
            log.error(f"  Batch {batch_num + 1} failed: {e}")
            stats['failed'] += len(batch)

    # Summary
    log.info("=" * 60)
    log.info("EMBED & STORE SUMMARY")
    log.info("=" * 60)
    log.info(f"  {'stored':<20} : {stats['stored']}")
    log.info(f"  {'failed':<20} : {stats['failed']}")
    log.info(f"  {'total in ChromaDB':<20} : {col.count()}")
    log.info(f"  {'db path':<20} : {os.path.abspath(args.db_path)}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
