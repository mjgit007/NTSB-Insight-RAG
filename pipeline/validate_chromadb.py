"""
validate_chromadb.py - Verify ChromaDB collection contents

Checks:
  1. Collection stats (count, sample IDs)
  2. Metadata completeness (no empty ntsb_no, section, state)
  3. Embedding sanity (correct dimensions, no zero vectors)
  4. Section distribution (how many chunks per section)
  5. NtsbNo distribution (how many chunks per report)
  6. Spot-check: fetch a specific NtsbNo and print its chunks
  7. Similarity search smoke test (query without LLM)

Usage:
  python validate_chromadb.py
  python validate_chromadb.py --ntsb-no ANC24FA029
"""

import os
import argparse
import chromadb

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectordb')
COLLECTION     = 'ntsb_reports'


def main():
    parser = argparse.ArgumentParser(description='Validate ChromaDB NTSB collection')
    parser.add_argument('--ntsb-no', default=None, help='Spot-check a specific NtsbNo')
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col    = client.get_collection(COLLECTION)

    print("=" * 60)
    print("CHECK 1: Collection Stats")
    print("=" * 60)
    total = col.count()
    print(f"  Total chunks     : {total}")
    peek = col.peek(limit=3)
    print(f"  Sample IDs       : {peek['ids']}")
    print(f"  Embedding dim    : {len(peek['embeddings'][0])}")
    print()

    # Fetch all chunks (ids + metadata + embeddings)
    print("=" * 60)
    print("CHECK 2: Metadata Completeness")
    print("=" * 60)
    all_data = col.get(include=['metadatas', 'embeddings'])
    ids       = all_data['ids']
    metadatas = all_data['metadatas']
    embeddings = all_data['embeddings']

    missing_ntsb    = [i for i, m in zip(ids, metadatas) if not m.get('ntsb_no')]
    missing_section = [i for i, m in zip(ids, metadatas) if not m.get('section')]
    missing_state   = [i for i, m in zip(ids, metadatas) if not m.get('state')]
    missing_make    = [i for i, m in zip(ids, metadatas) if not m.get('make')]

    print(f"  Missing ntsb_no  : {len(missing_ntsb)}")
    print(f"  Missing section  : {len(missing_section)}")
    print(f"  Missing state    : {len(missing_state)}")
    print(f"  Missing make     : {len(missing_make)}")
    if missing_ntsb:
        print(f"  Offending IDs    : {missing_ntsb[:5]}")
    print()

    print("=" * 60)
    print("CHECK 3: Embedding Sanity")
    print("=" * 60)
    dims        = [len(e) for e in embeddings]
    zero_vecs   = [ids[i] for i, e in enumerate(embeddings) if all(v == 0.0 for v in e)]
    unique_dims = set(dims)
    print(f"  Embedding dims   : {unique_dims}")
    print(f"  Zero vectors     : {len(zero_vecs)}")
    print(f"  Min norm (sample): {min(sum(v**2 for v in e)**0.5 for e in embeddings[:10]):.4f}")
    print(f"  Max norm (sample): {max(sum(v**2 for v in e)**0.5 for e in embeddings[:10]):.4f}")
    if zero_vecs:
        print(f"  Zero vector IDs  : {zero_vecs[:5]}")
    print()

    print("=" * 60)
    print("CHECK 4: Section Distribution")
    print("=" * 60)
    section_counts = {}
    for m in metadatas:
        s = m.get('section', 'unknown')
        section_counts[s] = section_counts.get(s, 0) + 1
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        bar = '#' * (count * 30 // max(section_counts.values()))
        print(f"  {section:<40} : {count:>4}  {bar}")
    print()

    print("=" * 60)
    print("CHECK 5: NtsbNo Distribution")
    print("=" * 60)
    ntsb_counts = {}
    for m in metadatas:
        n = m.get('ntsb_no', 'unknown')
        ntsb_counts[n] = ntsb_counts.get(n, 0) + 1
    unique_reports = len(ntsb_counts)
    avg_chunks     = total / unique_reports if unique_reports else 0
    min_chunks     = min(ntsb_counts.values())
    max_chunks     = max(ntsb_counts.values())
    print(f"  Unique reports   : {unique_reports}")
    print(f"  Avg chunks/report: {avg_chunks:.1f}")
    print(f"  Min chunks/report: {min_chunks}")
    print(f"  Max chunks/report: {max_chunks}")
    print()

    print("=" * 60)
    print("CHECK 6: Spot-check by NtsbNo")
    print("=" * 60)
    target = args.ntsb_no or list(ntsb_counts.keys())[0]
    result = col.get(
        where={'ntsb_no': target},
        include=['metadatas', 'documents']
    )
    print(f"  NtsbNo           : {target}")
    print(f"  Chunks found     : {len(result['ids'])}")
    for cid, meta, doc in zip(result['ids'], result['metadatas'], result['documents']):
        print(f"\n  --- {cid} ---")
        print(f"  Section : {meta.get('section')}")
        print(f"  State   : {meta.get('state')}  |  Make: {meta.get('make')}  |  Injury: {meta.get('injury_severity')}")
        print(f"  Text    : {doc[:200]}...")
    print()

    print("=" * 60)
    print("CHECK 7: Similarity Search Smoke Test")
    print("=" * 60)
    print("  (Querying with a raw text — no LLM, just vector similarity)")

    # Load .env for API key
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("  GOOGLE_API_KEY not found — skipping similarity test.")
        return

    from google import genai
    from google.genai import types

    gemini   = genai.Client(api_key=api_key)
    query    = "pilot lost control during landing due to crosswind"
    response = gemini.models.embed_content(
        model='gemini-embedding-001',
        contents=[query],
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
    )
    query_embedding = response.embeddings[0].values

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=['metadatas', 'documents', 'distances']
    )

    print(f"  Query            : '{query}'")
    print(f"  Top 3 results:")
    for i, (cid, meta, doc, dist) in enumerate(zip(
        results['ids'][0],
        results['metadatas'][0],
        results['documents'][0],
        results['distances'][0]
    ), 1):
        print(f"\n  [{i}] {cid}  (distance={dist:.4f})")
        print(f"      Section  : {meta.get('section')}")
        print(f"      NtsbNo   : {meta.get('ntsb_no')} | State: {meta.get('state')} | Make: {meta.get('make')}")
        print(f"      Text     : {doc[:200]}...")

    print()
    print("=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
