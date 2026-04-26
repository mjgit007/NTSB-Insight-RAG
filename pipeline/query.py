"""
query.py - Query the NTSB RAG system

Flow:
  1. Embed the user query with Gemini (RETRIEVAL_QUERY task type)
  2. Optional: apply metadata pre-filters (state, make, weather, injury, etc.)
  3. Vector similarity search in ChromaDB
  4. Send top-k chunks as context to Gemini Flash for answer generation

Usage:
  python query.py --query "What causes most fatal accidents in IMC conditions?"
  python query.py --query "Cessna crashes in Florida" --state Florida --make Cessna
  python query.py --query "Landing accidents" --injury Fatal --top-k 5
  python query.py --interactive
"""

import os
import argparse
import logging
import json

import chromadb
from google import genai
from google.genai import types

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
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'vectordb')
COLLECTION     = 'ntsb_reports'
EMBED_MODEL    = 'gemini-embedding-001'
ANSWER_MODEL   = 'gemini-2.5-flash'     # free tier, used for answer generation
TOP_K          = 5      # number of chunks to retrieve

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
# Step 1: Embed the query using Gemini
# Uses RETRIEVAL_QUERY task type (different from RETRIEVAL_DOCUMENT used at ingest)
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
# ChromaDB filter operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or
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
# Step 3: Search ChromaDB
# ---------------------------------------------------------------------------
def search(col, query_embedding: list[float], where_filter: dict | None, top_k: int) -> dict:
    kwargs = {
        'query_embeddings': [query_embedding],
        'n_results'        : top_k,
        'include'          : ['metadatas', 'documents', 'distances']
    }
    if where_filter:
        kwargs['where'] = where_filter

    return col.query(**kwargs)


# ---------------------------------------------------------------------------
# Step 4: Format retrieved chunks as context for Claude
# ---------------------------------------------------------------------------
def format_context(results: dict) -> str:
    chunks = []
    for cid, meta, doc, dist in zip(
        results['ids'][0],
        results['metadatas'][0],
        results['documents'][0],
        results['distances'][0]
    ):
        chunk = (
            f"--- Report: {meta.get('ntsb_no')} | Section: {meta.get('section')} "
            f"| State: {meta.get('state')} | Make: {meta.get('make')} "
            f"| Injury: {meta.get('injury_severity')} | Weather: {meta.get('weather')} "
            f"| Date: {meta.get('event_date', '')[:10]} | Similarity: {1 - dist:.3f} ---\n"
            f"{doc}\n"
        )
        chunks.append(chunk)
    return '\n'.join(chunks)


# ---------------------------------------------------------------------------
# Step 5: Generate answer with Gemini Flash (free tier)
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
def run_query(query_text: str, args, col, gemini_client):
    log.info(f"Query: {query_text}")

    # Embed query
    query_embedding = embed_query(gemini_client, query_text)
    log.info(f"Query embedded ({len(query_embedding)} dims)")

    # Build metadata filter
    where_filter = build_filter(args)
    if where_filter:
        log.info(f"Metadata filter: {json.dumps(where_filter)}")
    else:
        log.info("No metadata filter — searching full collection")

    # Search ChromaDB
    results = search(col, query_embedding, where_filter, args.top_k)
    retrieved = len(results['ids'][0])
    log.info(f"Retrieved {retrieved} chunks")

    if retrieved == 0:
        print("\nNo matching chunks found. Try relaxing your filters.")
        return

    # Show retrieved chunks summary
    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS")
    print("=" * 60)
    for i, (cid, meta, dist) in enumerate(zip(
        results['ids'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"  [{i}] {cid}")
        print(f"       Section : {meta.get('section')}")
        print(f"       Report  : {meta.get('ntsb_no')} | {meta.get('make')} | {meta.get('state')} | {meta.get('injury_severity')}")
        print(f"       Score   : {1 - dist:.3f}")
        print()

    # Format context
    context = format_context(results)

    # Ask Gemini
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
    parser.add_argument('--top-k',       type=int, default=TOP_K, help=f'Chunks to retrieve (default: {TOP_K})')

    # Metadata filter flags
    parser.add_argument('--state',   default=None, help='Filter by state (e.g. Florida)')
    parser.add_argument('--make',    default=None, help='Filter by aircraft make (e.g. Cessna)')
    parser.add_argument('--weather', default=None, help='Filter by weather: VMC or IMC')
    parser.add_argument('--injury',  default=None, help='Filter by injury: Fatal, Serious, Minor, None')
    parser.add_argument('--section', default=None, help='Filter by section (e.g. Analysis)')
    parser.add_argument('--ntsb-no', default=None, help='Filter by specific NtsbNo')
    args = parser.parse_args()

    load_env()

    # Validate API keys
    google_key = os.environ.get('GOOGLE_API_KEY')
    if not google_key:
        log.error("GOOGLE_API_KEY not set in .env")
        return

    # Init Gemini client (used for both embedding and answer generation)
    gemini_client = genai.Client(api_key=google_key)
    log.info(f"Answer model: {ANSWER_MODEL}")

    # Init ChromaDB
    chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col    = chroma.get_collection(COLLECTION)
    log.info(f"ChromaDB ready | Collection: '{COLLECTION}' | Chunks: {col.count()}")

    if args.interactive:
        print("\nNTSB RAG - Interactive Mode")
        print("Type your question and press Enter. Type 'exit' to quit.")
        print("Note: metadata filters from CLI args apply to all queries in this session.\n")
        while True:
            try:
                query_text = input("Question: ").strip()
                if query_text.lower() in ('exit', 'quit', 'q'):
                    break
                if not query_text:
                    continue
                run_query(query_text, args, col, gemini_client)
                print()
            except KeyboardInterrupt:
                break
        print("Goodbye.")

    elif args.query:
        run_query(args.query, args, col, gemini_client)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
