"""
setup_chromadb.py - Initialize and inspect ChromaDB for NTSB RAG

Usage:
  python setup_chromadb.py           # create collection and show info
  python setup_chromadb.py --reset   # drop and recreate collection (WARNING: deletes all data)
  python setup_chromadb.py --info    # show collection stats only
"""

import os
import argparse
import chromadb

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectordb')
COLLECTION     = 'ntsb_reports'

def main():
    parser = argparse.ArgumentParser(description='ChromaDB setup for NTSB RAG')
    parser.add_argument('--reset', action='store_true', help='Drop and recreate the collection')
    parser.add_argument('--info',  action='store_true', help='Show collection info only')
    args = parser.parse_args()

    # Create db directory if not exists
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    abs_path = os.path.abspath(CHROMA_DB_PATH)

    # Connect to persistent ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    if args.info:
        # Just show current state
        _print_info(client, abs_path)
        return

    if args.reset:
        # Drop existing collection
        existing = [c.name for c in client.list_collections()]
        if COLLECTION in existing:
            client.delete_collection(COLLECTION)
            print(f"Dropped collection: '{COLLECTION}'")
        else:
            print(f"Collection '{COLLECTION}' did not exist, nothing to drop.")

    # Create (or get existing) collection
    # hnsw:space=cosine -> cosine similarity scoring (best for semantic search)
    col = client.get_or_create_collection(
        name=COLLECTION,
        metadata={'hnsw:space': 'cosine'}
    )
    print(f"Collection '{COLLECTION}' ready.")
    _print_info(client, abs_path)


def _print_info(client: chromadb.PersistentClient, abs_path: str):
    print()
    print("=" * 50)
    print("ChromaDB Info")
    print("=" * 50)
    print(f"  DB path       : {abs_path}")

    collections = client.list_collections()
    print(f"  Collections   : {len(collections)}")
    print()

    for col in collections:
        c = client.get_collection(col.name)
        print(f"  Collection    : {c.name}")
        print(f"  Chunk count   : {c.count()}")
        print(f"  Similarity    : {(c.metadata or {}).get('hnsw:space', 'cosine')}")

        # Peek at first 3 stored chunks
        if c.count() > 0:
            peek = c.peek(limit=3)
            print(f"  Sample IDs    : {peek['ids']}")
            if peek['metadatas']:
                sample_meta = peek['metadatas'][0]
                print(f"  Sample meta   : ntsb_no={sample_meta.get('ntsb_no')}, "
                      f"section={sample_meta.get('section')}, "
                      f"state={sample_meta.get('state')}, "
                      f"make={sample_meta.get('make')}")
        print()
    print("=" * 50)


if __name__ == '__main__':
    main()
