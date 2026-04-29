"""
evaluate.py - RAGAS evaluation for the NTSB RAG pipeline

Measures pipeline quality across 4 metrics:
  - Faithfulness:       Does the answer stay grounded in retrieved chunks?
  - Answer Relevance:   Does the answer actually address the question asked?
  - Context Precision:  Are the retrieved chunks relevant to the question?
  - Context Recall:     Do the retrieved chunks contain the ground truth info?

Usage:
  python pipeline/evaluate.py                        # full eval, all 15 questions
  python pipeline/evaluate.py --subset 5             # quick run, first 5 questions
  python pipeline/evaluate.py --no-expand --no-rerank  # eval a weaker pipeline config
  python pipeline/evaluate.py --out results/eval_baseline.json  # save results

Requires:
  pip install ragas
  GOOGLE_API_KEY and COHERE_API_KEY set in .env
  (No OpenAI key needed — RAGAS scoring uses Gemini natively)
"""

import os
import re
import sys
import json
import argparse
import logging
from datetime import datetime
from types import SimpleNamespace

import chromadb
import cohere
from google import genai
from rank_bm25 import BM25Okapi

from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings

# Import pipeline functions from query.py
sys.path.insert(0, os.path.dirname(__file__))
from query import (
    load_env, build_bm25_index, embed_query, expand_query,
    build_filter, bm25_search, vector_search, reciprocal_rank_fusion,
    rerank as cohere_rerank, format_context,
    CHROMA_DB_PATH, COLLECTION, CANDIDATE_K, RERANK_TOP_N,
    RERANK_SCORE_CUTOFF, RRF_SCORE_CUTOFF, RRF_K,
    RERANK_MODEL, ANSWER_MODEL
)

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
# Golden dataset — 15 questions with ground truth answers
# Mix of: specific report lookups, pattern queries, edge cases
# Ground truth written from actual NTSB report content
# ---------------------------------------------------------------------------
GOLDEN_DATASET = [
    # --- Specific report lookups ---
    {
        "question": "What type of aircraft was involved in accident ERA22LA175 and what was the defining event?",
        "ground_truth": "A Learjet 45 operated by Learjet Inc was involved. The defining event was loss of control on the ground at Morristown, New Jersey, resulting in 4 minor injuries and substantial aircraft damage."
    },
    {
        "question": "What was the probable cause of accident CEN22LA359?",
        "ground_truth": "The pilot improperly monitored the operating environment which resulted in a collision with a tree located in the middle of a field. The NTSB finding was Personnel issues Monitoring environment - Pilot."
    },
    {
        "question": "What were the weather conditions during accident CEN22LA359?",
        "ground_truth": "The accident occurred in VMC (Visual Meteorological Conditions). The aircraft was an Air Tractor Inc operated in Louisiana resulting in minor injuries."
    },

    # --- Pattern queries ---
    {
        "question": "What are common causes of spatial disorientation accidents in IMC conditions?",
        "ground_truth": "Non-instrument-rated pilots continuing VFR flight into IMC conditions leading to spatial disorientation and loss of aircraft control. Common factors include pilot decision to fly into instrument meteorological conditions, inability to maintain aircraft attitude without visual references, and subsequent loss of control resulting in collision with terrain."
    },
    {
        "question": "What mistakes do pilots typically make when flying into clouds?",
        "ground_truth": "Pilots commonly make the decision to continue VFR flight into IMC conditions despite lacking instrument ratings or proficiency. This leads to spatial disorientation — inability to perceive aircraft attitude without visual cues — resulting in loss of control, uncontrolled descents, and high-energy terrain impact."
    },
    {
        "question": "What are typical findings in fatal accidents involving loss of control in flight?",
        "ground_truth": "Typical findings include spatial disorientation, failure to maintain aircraft control, inadvertent flight into IMC, pilot decision-making errors, and environmental factors such as clouds and low ceilings contributing to the outcome."
    },

    # --- Aircraft/make specific ---
    {
        "question": "What types of accidents commonly involve Cessna aircraft?",
        "ground_truth": "Cessna aircraft are commonly involved in accidents including runway excursions, hard landings, loss of control during takeoff or landing, fuel exhaustion, and inadvertent IMC encounters. These typically occur under Part 91 general aviation operations."
    },
    {
        "question": "What kinds of accidents happen during the landing phase of flight?",
        "ground_truth": "Landing phase accidents commonly involve runway excursions, hard landings, gear collapses, loss of directional control, bounced landings, and collisions with obstacles during approach. Contributing factors include crosswind conditions, improper airspeed management, and pilot technique errors."
    },

    # --- Weather specific ---
    {
        "question": "What role does carburetor ice play in engine failures?",
        "ground_truth": "Carburetor ice forms when moist air passes through the carburetor venturi causing temperature drops that freeze moisture. This restricts airflow and causes partial or total engine power loss. It typically occurs at lower power settings and in humid conditions, and can be prevented with carburetor heat application."
    },
    {
        "question": "How do fuel exhaustion accidents typically occur?",
        "ground_truth": "Fuel exhaustion accidents occur when pilots fail to properly plan fuel requirements, mismanage fuel systems, or misjudge fuel remaining. The result is total loss of engine power requiring forced landing, often with the aircraft sustaining substantial damage."
    },

    # --- Section-specific queries ---
    {
        "question": "What analysis did the NTSB provide for accident CEN22LA359?",
        "ground_truth": "The NTSB analysis for CEN22LA359 determined that the pilot improperly monitored the operating environment while applying insecticide, resulting in collision with a tree in the middle of a field. The pilot reported hearing a loud bang and later realized the aircraft had struck a tree."
    },
    {
        "question": "What pilot information is relevant to understanding IMC accidents?",
        "ground_truth": "Relevant pilot factors include instrument rating status (or lack thereof), total flight hours, instrument flight hours, recency of experience in IMC conditions, and proficiency in attitude instrument flying. Non-instrument-rated pilots who encounter IMC are at significantly elevated accident risk."
    },

    # --- Edge cases ---
    {
        "question": "What NTSB reports involve Boeing aircraft accidents?",
        "ground_truth": "NTSB reports involving Boeing aircraft include incidents such as engine fires, runway excursions, and in-flight emergencies. These typically involve commercial operations under Part 121 or Part 135 regulations."
    },
    {
        "question": "What are the probable causes of accidents in Alaska?",
        "ground_truth": "Accidents in Alaska frequently involve challenging terrain, adverse weather conditions including low visibility and icing, fuel exhaustion due to remote operations, controlled flight into terrain (CFIT), and loss of control. The remote operating environment with few alternate airports increases risk."
    },
    {
        "question": "What findings are typical in accidents where pilots failed to maintain altitude?",
        "ground_truth": "Typical findings include aircraft altitude not attained or maintained, spatial disorientation, personnel issues with aircraft control, environmental factors such as clouds or night conditions contributing to the outcome, and in some cases mechanical factors affecting aircraft performance."
    },
]

# ---------------------------------------------------------------------------
# Run a single question through the full pipeline
# Returns: answer (str), contexts (list[str])
# ---------------------------------------------------------------------------
def run_pipeline(
    question: str,
    col,
    gemini_client,
    cohere_client,
    bm25_index,
    args: SimpleNamespace
) -> tuple[str, list[str]]:

    # Query expansion
    use_expand = not args.no_expand
    if use_expand:
        expanded = expand_query(gemini_client, question)
        retrieval_query = expanded if expanded and expanded != question else question
    else:
        retrieval_query = question

    # Embed
    query_embedding = embed_query(gemini_client, retrieval_query)

    # Build filter (no metadata filters in eval — broad search)
    where_filter = None

    use_hybrid = not args.no_hybrid
    use_rerank = not args.no_rerank and cohere_client is not None

    if use_hybrid:
        bm25, all_ids, all_metadatas, all_documents = bm25_index

        bm25_results = bm25_search(
            retrieval_query, bm25, all_ids, all_metadatas, all_documents,
            where_filter, CANDIDATE_K
        )
        vector_results = vector_search(col, query_embedding, where_filter, CANDIDATE_K)

        rrf_top = RERANK_TOP_N if use_rerank else args.top_k
        fused = reciprocal_rank_fusion(bm25_results, vector_results, rrf_top)

        if use_rerank and fused:
            final_chunks = cohere_rerank(cohere_client, retrieval_query, fused, args.top_k)

            # Score cutoff
            if final_chunks and final_chunks[0]['rerank_score'] < RERANK_SCORE_CUTOFF:
                return "No relevant reports found for this query.", []
        else:
            final_chunks = fused[:args.top_k]
            if final_chunks and final_chunks[0]['rrf_score'] < RRF_SCORE_CUTOFF:
                return "No relevant reports found for this query.", []
    else:
        vector_results = vector_search(col, query_embedding, where_filter, args.top_k)
        ids   = vector_results['ids'][0]
        metas = vector_results['metadatas'][0]
        docs  = vector_results['documents'][0]
        dists = vector_results['distances'][0]
        final_chunks = [
            {'id': cid, 'metadata': meta, 'document': doc,
             'rrf_score': 1 - dist, 'bm25_score': 0, 'vector_score': 1 - dist}
            for cid, meta, doc, dist in zip(ids, metas, docs, dists)
        ]

    if not final_chunks:
        return "No relevant reports found.", []

    contexts = [c['document'] for c in final_chunks]
    context_str = format_context(final_chunks)

    # Generate answer
    from query import SYSTEM_PROMPT
    prompt = f"""{SYSTEM_PROMPT}

Based on the following NTSB accident report excerpts, answer this question:

Question: {question}

Context:
{context_str}
"""
    response = gemini_client.models.generate_content(model=ANSWER_MODEL, contents=prompt)
    answer = response.text.strip()

    return answer, contexts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate NTSB RAG pipeline with RAGAS')
    parser.add_argument('--subset',    type=int, default=None, help='Run only first N questions (default: all)')
    parser.add_argument('--top-k',     type=int, default=5,    help='Chunks per query (default: 5)')
    parser.add_argument('--no-hybrid', action='store_true',    help='Disable BM25 hybrid search')
    parser.add_argument('--no-rerank', action='store_true',    help='Disable Cohere reranker')
    parser.add_argument('--no-expand', action='store_true',    help='Disable query expansion')
    parser.add_argument('--out',       default=None,           help='Save results to JSON file')
    args = parser.parse_args()

    load_env()

    google_key = os.environ.get('GOOGLE_API_KEY')
    cohere_key = os.environ.get('COHERE_API_KEY')

    if not google_key:
        log.error("GOOGLE_API_KEY not set in .env")
        return

    # Init pipeline clients
    gemini_client = genai.Client(api_key=google_key)
    cohere_client = cohere.ClientV2(api_key=cohere_key) if cohere_key and not args.no_rerank else None

    if not cohere_client:
        log.warning("Cohere reranker disabled")

    # Init ChromaDB
    chroma = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    col    = chroma.get_collection(COLLECTION)
    log.info(f"ChromaDB ready | {col.count()} chunks")

    # Build BM25 index
    bm25_index = None
    if not args.no_hybrid:
        bm25_index = build_bm25_index(col)

    # Select questions
    questions = GOLDEN_DATASET[:args.subset] if args.subset else GOLDEN_DATASET
    log.info(f"Running eval on {len(questions)} questions...")

    # Init RAGAS scorers — AsyncInstructor-wrapped OpenAI client → Gemini backend
    # instructor.from_openai(AsyncOpenAI(...)) produces AsyncInstructor which RAGAS
    # detects as async-capable and routes through agenerate() correctly.
    log.info("Initialising RAGAS scorers (gemini-2.5-flash via instructor+AsyncOpenAI)...")
    import instructor
    from openai import AsyncOpenAI
    async_openai = AsyncOpenAI(
        api_key  = google_key,
        base_url = 'https://generativelanguage.googleapis.com/v1beta/openai/'
    )
    # JSON mode avoids Gemini's OpenAI-compat layer rejecting multi-param tool calls
    async_instructor_client = instructor.from_openai(async_openai, mode=instructor.Mode.JSON)
    ragas_llm   = llm_factory('gemini-2.5-flash', client=async_instructor_client, adapter='litellm')
    ragas_embed = GoogleEmbeddings(model='gemini-embedding-001', client=gemini_client)

    faith_scorer  = Faithfulness(llm=ragas_llm)
    relev_scorer  = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embed)
    prec_scorer   = ContextPrecision(llm=ragas_llm)
    recall_scorer = ContextRecall(llm=ragas_llm)

    # Run pipeline + score each question
    results = []
    for i, item in enumerate(questions, 1):
        q  = item['question']
        gt = item['ground_truth']
        log.info(f"[{i}/{len(questions)}] {q[:70]}...")

        try:
            answer, contexts = run_pipeline(q, col, gemini_client, cohere_client, bm25_index, args)
        except Exception as e:
            log.error(f"  Pipeline failed: {e}")
            answer   = f"ERROR: {e}"
            contexts = []

        log.info(f"  Answer: {answer[:100]}...")

        # Score — RAGAS 0.4.x per-sample async API
        import asyncio
        scores = {'faithfulness': None, 'answer_relevancy': None,
                  'context_precision': None, 'context_recall': None}
        if contexts:
            try:
                scores['faithfulness'] = asyncio.run(faith_scorer.ascore(
                    user_input=q, response=answer, retrieved_contexts=contexts
                )).value
            except Exception as e:
                log.warning(f"  faithfulness scoring failed: {e}")

            try:
                scores['answer_relevancy'] = asyncio.run(relev_scorer.ascore(
                    user_input=q, response=answer
                )).value
            except Exception as e:
                log.warning(f"  answer_relevancy scoring failed: {e}")

            try:
                scores['context_precision'] = asyncio.run(prec_scorer.ascore(
                    user_input=q, reference=gt, retrieved_contexts=contexts
                )).value
            except Exception as e:
                log.warning(f"  context_precision scoring failed: {e}")

            try:
                scores['context_recall'] = asyncio.run(recall_scorer.ascore(
                    user_input=q, retrieved_contexts=contexts, reference=gt
                )).value
            except Exception as e:
                log.warning(f"  context_recall scoring failed: {e}")

        results.append({
            'question':          q,
            'answer':            answer,
            'contexts':          contexts,
            'reference':         gt,
            **scores,
        })
        log.info(f"  Scores: faith={scores['faithfulness']} relev={scores['answer_relevancy']} "
                 f"prec={scores['context_precision']} recall={scores['context_recall']}")

    # Aggregate
    def _mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else float('nan')

    faith_avg  = _mean('faithfulness')
    relev_avg  = _mean('answer_relevancy')
    prec_avg   = _mean('context_precision')
    recall_avg = _mean('context_recall')

    # Print results
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  {'Faithfulness':<25} : {faith_avg:.4f}")
    print(f"  {'Answer Relevancy':<25} : {relev_avg:.4f}")
    print(f"  {'Context Precision':<25} : {prec_avg:.4f}")
    print(f"  {'Context Recall':<25} : {recall_avg:.4f}")
    print(f"\n  {'Questions evaluated':<25} : {len(results)}")
    print(f"  {'Pipeline config':<25} : "
          f"hybrid={'off' if args.no_hybrid else 'on'} "
          f"rerank={'off' if args.no_rerank else 'on'} "
          f"expand={'off' if args.no_expand else 'on'}")
    print("=" * 60)

    # Per-question breakdown
    print("\nPER-QUESTION SCORES:")
    print(f"{'#':<4} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Recall':>6}  Question")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        def _fmt(v): return f"{v:6.3f}" if v is not None else "   n/a"
        print(f"{i:<4} {_fmt(r['faithfulness'])} {_fmt(r['answer_relevancy'])} "
              f"{_fmt(r['context_precision'])} {_fmt(r['context_recall'])}  "
              f"{r['question'][:55]}")

    # Save results
    if args.out:
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        output = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'hybrid': not args.no_hybrid,
                'rerank': not args.no_rerank,
                'expand': not args.no_expand,
                'top_k':  args.top_k,
            },
            'scores': {
                'faithfulness':      faith_avg,
                'answer_relevancy':  relev_avg,
                'context_precision': prec_avg,
                'context_recall':    recall_avg,
            },
            'per_question': results,
        }
        with open(args.out, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        log.info(f"Results saved to {args.out}")


if __name__ == '__main__':
    main()
