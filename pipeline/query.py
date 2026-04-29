"""
query.py - Query the NTSB RAG system

Flow:
  1. Expand the query with NTSB aviation terminology via Gemini Flash
  2. Embed the expanded query with Gemini (RETRIEVAL_QUERY task type)
  3. Optional: apply metadata pre-filters (state, make, weather, injury, etc.)
  4. Hybrid search: BM25 keyword + vector cosine similarity, merged via RRF (top 20)
  5. Rerank top-20 candidates with Cohere rerank-v3.5 → top 5
  6. Score cutoff — skip LLM if best score below threshold
  7. Send top-5 chunks as context to Gemini Flash for answer generation

Usage:
  python query.py --query "What causes most fatal accidents in IMC conditions?"
  python query.py --query "Cessna crashes in Florida" --state Florida --make Cessna
  python query.py --query "Landing accidents" --injury Fatal --top-k 5
  python query.py --interactive
  python query.py --query "ERA22LA175" --no-hybrid      # cosine-only mode
  python query.py --query "ERA22LA175" --no-rerank      # hybrid but skip reranker
  python query.py --query "pilot confused in clouds" --no-expand  # skip query expansion
"""

import os
import re
import argparse
import logging
import json

import chromadb
import cohere
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

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
CHROMA_DB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'vectordb')
COLLECTION      = 'ntsb_reports'
EMBED_MODEL     = 'gemini-embedding-001'
ANSWER_MODEL    = 'gemini-2.5-flash'
RERANK_MODEL    = 'rerank-v3.5'
TOP_K                = 5       # final chunks sent to LLM
CANDIDATE_K          = 50      # candidates fetched from each retriever before RRF
RERANK_TOP_N         = 20      # candidates passed to Cohere reranker
RRF_K                = 60      # RRF smoothing constant
RERANK_SCORE_CUTOFF  = 0.10    # minimum rerank score — below this, skip LLM call
RRF_SCORE_CUTOFF     = 0.015   # minimum RRF score when reranker is disabled

EXPAND_PROMPT = """You are an NTSB aviation safety expert. Your task is to rewrite a user query using precise NTSB terminology so it matches the language used in official accident investigation reports.

Rules:
- Output ONLY the rewritten query — no explanation, no preamble, no quotes
- Keep it concise (one sentence, max 30 words)
- Preserve specific identifiers (report numbers, aircraft registrations) exactly as given
- Add NTSB formal terms: spatial disorientation, inadvertent IMC, VFR-into-IMC, controlled flight into terrain (CFIT), loss of control in-flight (LOC-I), fuel exhaustion, carburetor ice, aerodynamic stall, runway excursion, hard landing, etc.
- If the query already uses NTSB terminology, return it unchanged

Examples:
  Input:  pilot confused in clouds lost control
  Output: spatial disorientation inadvertent IMC loss of aircraft control VFR-into-IMC

  Input:  ran out of fuel
  Output: fuel exhaustion total loss of engine power forced landing

  Input:  bad landing gear problem
  Output: landing gear malfunction collapse runway excursion

  Input:  ERA22LA175
  Output: ERA22LA175
"""

SYSTEM_PROMPT = """You are an aviation safety analyst with deep expertise in NTSB accident investigations.

You answer questions based strictly on NTSB accident investigation reports provided as context.

Guidelines:
- Base your answer only on the provided context chunks
- Always cite the NTSB report number (NtsbNo) when referencing specific accidents
- If multiple reports are relevant, summarize patterns across them
- If the context does not contain enough information, say so clearly
- Be factual and precise — this is safety-critical information
- When mentioning probable causes, quote directly from the report
"""

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# Step 1a: Query expansion — rewrite plain English into NTSB terminology
# ---------------------------------------------------------------------------
def expand_query(gemini_client: genai.Client, query_text: str) -> str:
    response = gemini_client.models.generate_content(
        model=ANSWER_MODEL,
        contents=f"{EXPAND_PROMPT}\n\nInput: {query_text}\nOutput:"
    )
    expanded = response.text.strip()
    return expanded


# ---------------------------------------------------------------------------
# Step 1b: Embed the query using Gemini
# ---------------------------------------------------------------------------
def embed_query(gemini_client: genai.Client, query_text: str) -> list[float]:
    response = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query_text],
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
    )
    return response.embeddings[0].values


# ---------------------------------------------------------------------------
# Step 2: Build ChromaDB where-filter from CLI args
# ---------------------------------------------------------------------------
def build_filter(args) -> dict | None:
    conditions = []

    if args.state:
        conditions.append({'state': {'$eq': args.state}})
    if args.make:
        conditions.append({'make': {'$eq': args.make.upper()}})
    if args.weather:
        conditions.append({'weather': {'$eq': args.weather.upper()}})
    if args.injury:
        conditions.append({'injury_severity': {'$eq': args.injury}})
    if args.section:
        conditions.append({'section': {'$eq': args.section}})
    if args.ntsb_no:
        conditions.append({'ntsb_no': {'$eq': args.ntsb_no}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {'$and': conditions}


# ---------------------------------------------------------------------------
# BM25 index — built once per session from the full ChromaDB collection
# ---------------------------------------------------------------------------
def build_bm25_index(col):
    all_ids, all_metadatas, all_documents = [], [], []
    total = col.count()
    log.info(f"Building BM25 index over {total} chunks...")

    for offset in range(0, total, 5000):
        batch = col.get(limit=5000, offset=offset, include=['metadatas', 'documents'])
        all_ids.extend(batch['ids'])
        all_metadatas.extend(batch['metadatas'])
        all_documents.extend(batch['documents'])

    tokenized = [re.findall(r'[a-z0-9]+', doc.lower()) for doc in all_documents]
    bm25 = BM25Okapi(tokenized)
    log.info(f"BM25 index ready ({len(all_ids)} chunks)")
    return bm25, all_ids, all_metadatas, all_documents


# ---------------------------------------------------------------------------
# Metadata filter matching for BM25 results
# ---------------------------------------------------------------------------
def _matches_filter(meta: dict, where_filter: dict) -> bool:
    if '$and' in where_filter:
        return all(_matches_filter(meta, c) for c in where_filter['$and'])
    if '$or' in where_filter:
        return any(_matches_filter(meta, c) for c in where_filter['$or'])
    for field, condition in where_filter.items():
        if isinstance(condition, dict):
            op, val = next(iter(condition.items()))
            actual = meta.get(field)
            if op == '$eq'  and actual != val:      return False
            if op == '$ne'  and actual == val:      return False
            if op == '$in'  and actual not in val:  return False
            if op == '$nin' and actual in val:      return False
        else:
            if meta.get(field) != condition:        return False
    return True


# ---------------------------------------------------------------------------
# BM25 keyword search
# ---------------------------------------------------------------------------
def bm25_search(
    query_text: str,
    bm25: BM25Okapi,
    all_ids: list,
    all_metadatas: list,
    all_documents: list,
    where_filter: dict | None,
    top_n: int
) -> list[tuple]:
    tokens = re.findall(r'[a-z0-9]+', query_text.lower())
    scores = bm25.get_scores(tokens)

    scored = [
        (cid, meta, doc, float(scores[idx]))
        for idx, (cid, meta, doc) in enumerate(zip(all_ids, all_metadatas, all_documents))
        if not where_filter or _matches_filter(meta, where_filter)
    ]
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Vector (cosine) search via ChromaDB
# ---------------------------------------------------------------------------
def vector_search(col, query_embedding: list[float], where_filter: dict | None, top_k: int) -> dict:
    kwargs = {
        'query_embeddings': [query_embedding],
        'n_results'        : top_k,
        'include'          : ['metadatas', 'documents', 'distances']
    }
    if where_filter:
        kwargs['where'] = where_filter
    return col.query(**kwargs)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion — merge BM25 + vector results
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    bm25_results: list[tuple],
    vector_results: dict,
    top_k: int
) -> list[dict]:
    rrf_scores  = {}
    chunk_store = {}

    for rank, (cid, meta, doc, bm25_score) in enumerate(bm25_results, start=1):
        rrf_scores.setdefault(cid, {
            'rrf': 0.0, 'bm25_rank': None, 'vector_rank': None,
            'bm25_score': 0.0, 'vector_score': 0.0
        })
        rrf_scores[cid]['rrf']        += 1.0 / (rank + RRF_K)
        rrf_scores[cid]['bm25_rank']   = rank
        rrf_scores[cid]['bm25_score']  = bm25_score
        chunk_store[cid] = {'metadata': meta, 'document': doc}

    for rank, (cid, meta, doc, dist) in enumerate(zip(
        vector_results['ids'][0],
        vector_results['metadatas'][0],
        vector_results['documents'][0],
        vector_results['distances'][0]
    ), start=1):
        cosine_sim = 1.0 - dist
        rrf_scores.setdefault(cid, {
            'rrf': 0.0, 'bm25_rank': None, 'vector_rank': None,
            'bm25_score': 0.0, 'vector_score': 0.0
        })
        rrf_scores[cid]['rrf']          += 1.0 / (rank + RRF_K)
        rrf_scores[cid]['vector_rank']   = rank
        rrf_scores[cid]['vector_score']  = cosine_sim
        chunk_store.setdefault(cid, {'metadata': meta, 'document': doc})

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1]['rrf'], reverse=True)

    results = []
    for cid, scores in ranked[:top_k]:
        results.append({
            'id':           cid,
            'metadata':     chunk_store[cid]['metadata'],
            'document':     chunk_store[cid]['document'],
            'rrf_score':    scores['rrf'],
            'bm25_rank':    scores['bm25_rank'],
            'vector_rank':  scores['vector_rank'],
            'bm25_score':   scores['bm25_score'],
            'vector_score': scores['vector_score'],
        })
    return results


# ---------------------------------------------------------------------------
# Cohere reranker — rescores query+chunk pairs together
# Returns top_k chunks re-ordered by relevance score
# ---------------------------------------------------------------------------
def rerank(cohere_client: cohere.Client, query: str, candidates: list[dict], top_k: int) -> list[dict]:
    documents = [c['document'] for c in candidates]

    response = cohere_client.rerank(
        model     = RERANK_MODEL,
        query     = query,
        documents = documents,
        top_n     = top_k,
    )

    reranked = []
    for hit in response.results:
        c = candidates[hit.index]
        reranked.append({
            **c,
            'rerank_score':    hit.relevance_score,
            'rerank_position': hit.index,         # original position before rerank
        })
    return reranked


# ---------------------------------------------------------------------------
# Format retrieved chunks as context for the LLM
# ---------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        meta = c['metadata']
        parts.append(
            f"--- Report: {meta.get('ntsb_no')} | Section: {meta.get('section')} "
            f"| State: {meta.get('state')} | Make: {meta.get('make')} "
            f"| Injury: {meta.get('injury_severity')} | Weather: {meta.get('weather')} "
            f"| Date: {meta.get('event_date', '')[:10]} ---\n"
            f"{c['document']}\n"
        )
    return '\n'.join(parts)


def format_context_cosine(results: dict) -> str:
    chunks = []
    for cid, meta, doc, dist in zip(
        results['ids'][0],
        results['metadatas'][0],
        results['documents'][0],
        results['distances'][0]
    ):
        chunks.append(
            f"--- Report: {meta.get('ntsb_no')} | Section: {meta.get('section')} "
            f"| State: {meta.get('state')} | Make: {meta.get('make')} "
            f"| Injury: {meta.get('injury_severity')} | Weather: {meta.get('weather')} "
            f"| Date: {meta.get('event_date', '')[:10]} | Similarity: {1 - dist:.3f} ---\n"
            f"{doc}\n"
        )
    return '\n'.join(chunks)


# ---------------------------------------------------------------------------
# Generate answer with Gemini Flash
# ---------------------------------------------------------------------------
def ask_gemini(gemini_client: genai.Client, query: str, context: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

Based on the following NTSB accident report excerpts, answer this question:

Question: {query}

Context:
{context}
"""
    response = gemini_client.models.generate_content(
        model=ANSWER_MODEL,
        contents=prompt
    )
    return response.text


# ---------------------------------------------------------------------------
# Single query run
# ---------------------------------------------------------------------------
def run_query(query_text: str, args, col, gemini_client, cohere_client=None, bm25_index=None):
    log.info(f"Query: {query_text}")

    # Query expansion — rewrite into NTSB terminology before retrieval
    use_expand = not getattr(args, 'no_expand', False)
    if use_expand:
        expanded_query = expand_query(gemini_client, query_text)
        if expanded_query and expanded_query != query_text:
            log.info(f"Expanded query: {expanded_query}")
            retrieval_query = expanded_query
        else:
            log.info("Query expansion unchanged — using original")
            retrieval_query = query_text
    else:
        retrieval_query = query_text

    query_embedding = embed_query(gemini_client, retrieval_query)
    log.info(f"Query embedded ({len(query_embedding)} dims)")

    where_filter = build_filter(args)
    if where_filter:
        log.info(f"Metadata filter: {json.dumps(where_filter)}")
    else:
        log.info("No metadata filter — searching full collection")

    use_hybrid     = not getattr(args, 'no_hybrid', False) and bm25_index is not None
    use_rerank     = not getattr(args, 'no_rerank', False) and cohere_client is not None
    rerank_cutoff  = args.score_cutoff if args.score_cutoff is not None else RERANK_SCORE_CUTOFF
    rrf_cutoff     = args.score_cutoff if args.score_cutoff is not None else RRF_SCORE_CUTOFF

    if use_hybrid:
        bm25, all_ids, all_metadatas, all_documents = bm25_index

        bm25_results = bm25_search(
            retrieval_query, bm25, all_ids, all_metadatas, all_documents,
            where_filter, CANDIDATE_K
        )
        log.info(f"BM25 returned {len(bm25_results)} candidates")

        vector_results = vector_search(col, query_embedding, where_filter, CANDIDATE_K)
        log.info(f"Vector search returned {len(vector_results['ids'][0])} candidates")

        # RRF merge — get RERANK_TOP_N candidates for reranker, or TOP_K if no reranker
        rrf_top = RERANK_TOP_N if use_rerank else args.top_k
        fused = reciprocal_rank_fusion(bm25_results, vector_results, rrf_top)
        log.info(f"RRF merged → {len(fused)} candidates")

        if use_rerank and len(fused) > 0:
            log.info(f"Reranking {len(fused)} candidates with Cohere {RERANK_MODEL}...")
            final_chunks = rerank(cohere_client, retrieval_query, fused, args.top_k)
            log.info(f"Reranked → {len(final_chunks)} final chunks")

            # Score cutoff — skip LLM if best rerank score is below threshold
            if final_chunks and final_chunks[0]['rerank_score'] < rerank_cutoff:
                log.info(f"Best rerank score {final_chunks[0]['rerank_score']:.4f} < {rerank_cutoff} cutoff — no relevant reports found")
                print(f"\nNo relevant reports found for this query (confidence too low: {final_chunks[0]['rerank_score']:.4f} < {rerank_cutoff}).")
                print("Try rephrasing, broadening your filters, or checking that the relevant PDFs are in the dataset.")
                return

            retrieval_mode = "Hybrid: BM25 + Vector via RRF → Cohere Rerank"
        else:
            final_chunks = fused[:args.top_k]

            # Score cutoff — skip LLM if best RRF score is below threshold
            if final_chunks and final_chunks[0]['rrf_score'] < rrf_cutoff:
                log.info(f"Best RRF score {final_chunks[0]['rrf_score']:.4f} < {rrf_cutoff} cutoff — no relevant reports found")
                print(f"\nNo relevant reports found for this query (confidence too low: {final_chunks[0]['rrf_score']:.4f} < {rrf_cutoff}).")
                print("Try rephrasing, broadening your filters, or checking that the relevant PDFs are in the dataset.")
                return

            retrieval_mode = "Hybrid: BM25 + Vector via RRF"

    else:
        # Cosine-only fallback
        results = vector_search(col, query_embedding, where_filter, args.top_k)
        retrieved = len(results['ids'][0])
        log.info(f"Retrieved {retrieved} chunks (cosine-only)")

        if retrieved == 0:
            print("\nNo matching chunks found. Try relaxing your filters.")
            return

        print("\n" + "=" * 60)
        print("RETRIEVED CHUNKS  [Cosine-only]")
        print("=" * 60)
        for i, (cid, meta, dist) in enumerate(zip(
            results['ids'][0], results['metadatas'][0], results['distances'][0]
        ), 1):
            print(f"  [{i}] {cid}")
            print(f"       Section : {meta.get('section')}")
            print(f"       Report  : {meta.get('ntsb_no')} | {meta.get('make')} | {meta.get('state')} | {meta.get('injury_severity')}")
            print(f"       Score   : {1 - dist:.3f}")
            print()

        context = format_context_cosine(results)
        print("=" * 60)
        print("GEMINI ANSWER")
        print("=" * 60)
        answer = ask_gemini(gemini_client, query_text, context)
        print(answer)
        print("=" * 60)
        return

    if not final_chunks:
        print("\nNo matching chunks found. Try relaxing your filters.")
        return

    print("\n" + "=" * 60)
    print(f"RETRIEVED CHUNKS  [{retrieval_mode}]")
    print("=" * 60)
    for i, c in enumerate(final_chunks, 1):
        meta = c['metadata']
        print(f"  [{i}] {c['id']}")
        print(f"       Section : {meta.get('section')}")
        print(f"       Report  : {meta.get('ntsb_no')} | {meta.get('make')} | {meta.get('state')} | {meta.get('injury_severity')}")
        if 'rerank_score' in c:
            print(f"       Rerank  : {c['rerank_score']:.4f}  "
                  f"(was RRF pos #{c['rerank_position'] + 1})  "
                  f"BM25={c['bm25_score']:.3f}  Vector={c['vector_score']:.3f}")
        else:
            print(f"       RRF     : {c['rrf_score']:.4f}  "
                  f"BM25={c['bm25_score']:.3f}(rank {c['bm25_rank']})  "
                  f"Vector={c['vector_score']:.3f}(rank {c['vector_rank']})")
        print()

    context = format_context(final_chunks)
    print("=" * 60)
    print("GEMINI ANSWER")
    print("=" * 60)
    answer = ask_gemini(gemini_client, query_text, context)
    print(answer)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Query the NTSB RAG system')
    parser.add_argument('--query',       default=None,  help='Question to ask')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--top-k',       type=int, default=TOP_K, help=f'Final chunks to return (default: {TOP_K})')
    parser.add_argument('--no-hybrid',      action='store_true', help='Use cosine-only search (skip BM25)')
    parser.add_argument('--no-rerank',      action='store_true', help='Skip Cohere reranker')
    parser.add_argument('--no-expand',      action='store_true', help='Skip query expansion')
    parser.add_argument('--score-cutoff',   type=float, default=None,
                        help=f'Override score cutoff (rerank default: {RERANK_SCORE_CUTOFF}, RRF default: {RRF_SCORE_CUTOFF}). Set 0 to disable.')

    # Metadata filter flags
    parser.add_argument('--state',   default=None, help='Filter by state (e.g. Florida)')
    parser.add_argument('--make',    default=None, help='Filter by aircraft make (e.g. Cessna)')
    parser.add_argument('--weather', default=None, help='Filter by weather: VMC or IMC')
    parser.add_argument('--injury',  default=None, help='Filter by injury: Fatal, Serious, Minor, None')
    parser.add_argument('--section', default=None, help='Filter by section (e.g. Analysis)')
    parser.add_argument('--ntsb-no', default=None, help='Filter by specific NtsbNo')
    args = parser.parse_args()

    load_env()

    google_key  = os.environ.get('GOOGLE_API_KEY')
    cohere_key  = os.environ.get('COHERE_API_KEY')

    if not google_key:
        log.error("GOOGLE_API_KEY not set in .env")
        return

    # Init Gemini client
    gemini_client = genai.Client(api_key=google_key)
    log.info(f"Answer model: {ANSWER_MODEL}")

    # Init Cohere client (optional — reranker skipped if key missing)
    cohere_client = None
    if cohere_key and not args.no_rerank:
        cohere_client = cohere.ClientV2(api_key=cohere_key)
        log.info(f"Cohere reranker ready: {RERANK_MODEL}")
    elif not cohere_key:
        log.warning("COHERE_API_KEY not set — reranker disabled. Add to .env to enable.")

    # Init ChromaDB
    chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col    = chroma.get_collection(COLLECTION)
    log.info(f"ChromaDB ready | Collection: '{COLLECTION}' | Chunks: {col.count()}")

    # Build BM25 index (unless cosine-only mode)
    bm25_index = None
    if not args.no_hybrid:
        bm25_index = build_bm25_index(col)

    if args.interactive:
        mode_label = "BM25 + Vector + Cohere Rerank" if cohere_client else "BM25 + Vector"
        print(f"\nNTSB RAG - Interactive Mode  [{mode_label}]")
        print("Type your question and press Enter. Type 'exit' to quit.")
        print("Note: metadata filters from CLI args apply to all queries in this session.\n")
        while True:
            try:
                query_text = input("Question: ").strip()
                if query_text.lower() in ('exit', 'quit', 'q'):
                    break
                if not query_text:
                    continue
                run_query(query_text, args, col, gemini_client, cohere_client, bm25_index)
                print()
            except KeyboardInterrupt:
                break
        print("Goodbye.")

    elif args.query:
        run_query(args.query, args, col, gemini_client, cohere_client, bm25_index)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
