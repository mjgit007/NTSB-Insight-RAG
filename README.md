# NTSB Insight RAG

A local Retrieval-Augmented Generation (RAG) system built on real NTSB aviation accident data.
Ask plain-English questions across thousands of official investigation reports and get answers grounded in actual findings.

> New to RAG? Read [RAG: Give Your AI a Better Memory](https://nanotechbytes.com/post/rag-give-your-ai-a-better-memory?type=tech-byte) first — this project is the hands-on implementation.

---

## What Is NTSB?

The **National Transportation Safety Board (NTSB)** is an independent U.S. federal agency that investigates civil aviation accidents. Every accident — roughly 1,300 per year — results in a public investigation report covering:

- What happened (History of Flight)
- Who was flying (Pilot Information)
- Weather conditions at the time
- Aircraft damage assessment
- Analysis of contributing factors
- Probable Cause and Findings

NTSB publishes both a structured CSV database (36 columns per accident) and a full narrative PDF for each completed investigation. This project uses both.

---

## What This Project Does

This project builds a local RAG pipeline on top of NTSB data so you can ask questions like:

- *"What caused most fatal accidents in Alaska in IMC conditions?"*
- *"What patterns appear in Cessna crashes during landing?"*
- *"Show me cases where carburetor ice was a probable cause"*

No model training. No cloud vector database. Everything runs locally.

---

## Pipeline Architecture

```
User query (plain English)
        │
        ▼
┌─────────────────────┐
│   Query Expansion   │  Gemini Flash rewrites query into NTSB formal terminology
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Embed Query       │  gemini-embedding-001 (RETRIEVAL_QUERY, 3072-dim)
└─────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
┌──────────────┐              ┌──────────────────┐
│  BM25 Search │              │  Vector Search   │
│ (keyword)    │              │  (cosine, ChromaDB)│
└──────────────┘              └──────────────────┘
        │                              │
        └──────────┬───────────────────┘
                   ▼
        ┌─────────────────────┐
        │  RRF Fusion (k=60)  │  Merges BM25 + vector rankings
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Cohere Reranker    │  rerank-v3.5 cross-encoder, top 20 → top 5
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Score Cutoff      │  Below 0.10 → "no relevant results"
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Gemini Answer     │  gemini-2.5-flash generates grounded answer
        └─────────────────────┘
```

---

## How the Data Was Extracted

### Step 1 — CSV export from NTSB

Downloaded the accident database CSV from the [NTSB query page](https://data.ntsb.gov/avdata) covering **2010–2026** (26,872 records, 36 columns per row).

Key columns used:
| Column | Description |
|---|---|
| `NtsbNo` | Unique report identifier (e.g. `ERA21FA001`) — used as filename |
| `Mkey` | Numeric key used by the NTSB PDF API |
| `EventDate` | Date of accident |
| `State`, `City` | Location |
| `Make`, `Model` | Aircraft |
| `HighestInjuryLevel` | Fatal / Serious / Minor / None |
| `WeatherCondition` | VMC or IMC |
| `ProbableCause` | Short summary from CSV |
| `ReportStatus` | Completed / In-work |

The full CSV was split into 5-year bands for manageable processing:

```
data/NTSB_2010_2014.csv  →  8,539 records
data/NTSB_2015_2019.csv  →  8,191 records
data/NTSB_2020_2024.csv  →  8,106 records
data/NTSB_2025_2026.csv  →  2,036 records
```

### Step 2 — PDF download via NTSB public API

Each completed investigation has a PDF report. Downloaded using the NTSB public API:

```
GET https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/{Mkey}/pdf
```

Note: the API requires the numeric `Mkey` field — not the human-readable `NtsbNo`. The `NtsbNo` is used only as the local filename.

Result: **958 PDFs** downloaded for the 2020–2024 band, saved to `pdfs/NTSB_2020_2024/`.

---

## Project Structure

```
ntsb-insight/
│
├── data/                        # NTSB CSV exports (committed)
│   ├── NTSB_2010_2014.csv
│   ├── NTSB_2015_2019.csv
│   ├── NTSB_2020_2024.csv
│   └── NTSB_2025_2026.csv
│
├── pdfs/                        # Downloaded PDFs — gitignored (too large)
│   └── NTSB_2020_2024/
│       └── ERA21FA001.pdf ...
│
├── scripts/                     # Data acquisition utilities
│   ├── download.py              # Download PDFs from NTSB API
│   └── split_csv.py             # Split full CSV into year bands
│
├── pipeline/                    # Core RAG pipeline
│   ├── ingest.py                # PDF → section chunks → JSONL
│   ├── embed_and_store.py       # JSONL → Gemini embeddings → ChromaDB
│   ├── query.py                 # Question → hybrid search → rerank → AI answer
│   ├── evaluate.py              # RAGAS evaluation framework (15-question golden dataset)
│   ├── setup_chromadb.py        # Create / inspect / reset vector DB
│   └── validate_chromadb.py     # Sanity checks on stored embeddings
│
├── chunks/                      # JSONL output from ingest.py — gitignored
├── vectordb/                    # ChromaDB persistence — gitignored
├── logs/                        # Download and run logs — gitignored
├── results/                     # RAGAS evaluation output JSON — gitignored
│
├── .env.example                 # API key template — copy to .env and fill in
├── requirements.txt
└── .gitignore
```

---

## Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com) API key (free) — for embeddings and answer generation
- A [Cohere](https://dashboard.cohere.com) API key (free tier available) — for reranking

---

## Setup

> **Important:** All commands below must be run from the repo root (`ntsb-insight/`), not from inside `pipeline/` or `scripts/`.

```bash
git clone https://github.com/mjgit007/NTSB-Insight-RAG.git
cd NTSB-Insight-RAG

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your API keys
cp .env.example .env
```

### `.env` file — required keys

Open `.env` and fill in both keys:

```
GOOGLE_API_KEY=your_google_ai_studio_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

| Key | Where to get it | Used for |
|---|---|---|
| `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com) → Get API key | Embeddings + answer generation + query expansion |
| `COHERE_API_KEY` | [Cohere Dashboard](https://dashboard.cohere.com) → API keys | Reranking (optional — pipeline runs without it, reranker disabled) |

> Without `COHERE_API_KEY` the pipeline still works — it falls back to RRF-ranked results without the reranker pass.

```bash
# Create the chunks folder (gitignored, so not in the repo)
mkdir -p chunks results
```

---

## Step-by-Step: Running the Pipeline

### 1. Download PDFs

```bash
python scripts/download.py --csv data/NTSB_2020_2024.csv
```

Downloads completed-report PDFs into `pdfs/NTSB_2020_2024/`. Supports resume — already-downloaded files are skipped. Rate-limited to 30 per batch with retry logic for 429s and server errors.

**Output:** `pdfs/NTSB_2020_2024/*.pdf` (one file per accident, named by NtsbNo)

---

### 2. Ingest — PDF → Chunks

```bash
# Test with 10 PDFs first
python pipeline/ingest.py \
  --pdf-dir pdfs/NTSB_2020_2024 \
  --csv     data/NTSB_2020_2024.csv \
  --out     chunks/chunks_2020_2024.jsonl \
  --limit   10

# Full run
python pipeline/ingest.py \
  --pdf-dir pdfs/NTSB_2020_2024 \
  --csv     data/NTSB_2020_2024.csv \
  --out     chunks/chunks_2020_2024.jsonl
```

Each PDF is split into named sections (History of Flight, Analysis, Probable Cause, etc.) rather than fixed-size blocks. Each section becomes a chunk with 29 metadata fields attached from the CSV row.

**Output:** `chunks/chunks_2020_2024.jsonl` — one JSON line per chunk

Test result: **5 PDFs → 52 chunks**, avg ~10 chunks per report.

---

### 3. Set Up ChromaDB

```bash
python pipeline/setup_chromadb.py          # create collection
python pipeline/setup_chromadb.py --info   # inspect stats
python pipeline/setup_chromadb.py --reset  # wipe and recreate
```

Creates a persistent local vector database at `vectordb/` using cosine similarity.

---

### 4. Embed and Store

```bash
python pipeline/embed_and_store.py --jsonl chunks/chunks_2020_2024.jsonl
```

Reads the JSONL, calls Gemini (`gemini-embedding-001`, 3072 dimensions) to embed each chunk, and upserts into ChromaDB. Supports resume — chunks already in ChromaDB are skipped.

Test result: **52 chunks embedded and stored in 1.3 seconds**.

---

### 5. Validate ChromaDB (optional)

```bash
python pipeline/validate_chromadb.py
```

Runs sanity checks — confirms chunk count, checks metadata fields, and samples a few chunks to verify dimensions and content.

---

### 6. Query

```bash
# Basic query
python pipeline/query.py --query "What caused the fatal accident in Alaska?"

# With metadata filters (hybrid retrieval)
python pipeline/query.py \
  --query "engine failure during cruise" \
  --state Alaska \
  --injury Fatal \
  --weather IMC

# Interactive mode
python pipeline/query.py --interactive

# Disable individual pipeline stages (for comparison / debugging)
python pipeline/query.py --query "ERA22LA175" --no-hybrid    # cosine-only, no BM25
python pipeline/query.py --query "ERA22LA175" --no-rerank    # skip Cohere reranker
python pipeline/query.py --query "pilot error crash" --no-expand  # skip query expansion
```

**Available filters:** `--state`, `--make`, `--weather` (VMC/IMC), `--injury` (Fatal/Serious/Minor/None), `--section`, `--ntsb-no`

---

### 7. Evaluate Pipeline Quality (optional)

```bash
# Quick smoke test — 1 question
python pipeline/evaluate.py --subset 1

# Full evaluation — all 15 questions, save results
python pipeline/evaluate.py --out results/eval_full_pipeline.json

# Baseline comparison — no improvements enabled
python pipeline/evaluate.py --no-hybrid --no-rerank --no-expand \
  --out results/eval_baseline.json
```

Runs the 15-question golden dataset through the pipeline and scores with RAGAS:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer stay grounded in retrieved chunks? (no hallucination) |
| **Answer Relevancy** | Does the answer address the question asked? |
| **Context Precision** | Are retrieved chunks relevant to the question? |
| **Context Recall** | Do retrieved chunks contain the ground truth information? |

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| PDF parser | `pdfplumber` | Layout-aware, handles NTSB column formatting |
| Chunking | Section-aware | Each chunk is semantically complete |
| Chunk size | 4,000 chars / 400 overlap | ~1,000 tokens, safe for embedding limits |
| Vector DB | ChromaDB (local) | No cloud, persistent, zero cost |
| Embed model | `gemini-embedding-001` (3072 dims) | Free tier, high-dimensional |
| Retrieval | BM25 + vector via RRF | Hybrid catches both exact terms and semantic matches |
| Reranker | Cohere `rerank-v3.5` | Cross-encoder reads query+chunk together for precise relevance |
| Query expansion | Gemini Flash → NTSB terminology | Bridges plain English to formal report language |
| Score cutoff | Rerank score < 0.10 | Blocks LLM call when no relevant context found |
| Answer model | Gemini 2.5 Flash | Free tier; fast; grounded answers |
| Evaluation | RAGAS + 15-question golden dataset | Quantitative quality measurement across 4 metrics |

---

## Query Results

### Positive result — specific, grounded answer

```
Question: what caused the engine failure in Alaska?

RETRIEVED CHUNKS  [Hybrid: BM25 + Vector via RRF]
  [1] ANC25LA013_chunk_2  |  Probable Cause and Findings  |  CESSNA | Hawaii
  [2] ANC25LA013_chunk_1  |  Analysis                     |  CESSNA | Hawaii
  [3] ANC25LA013_chunk_4  |  History of Flight             |  CESSNA | Hawaii
  [4] ANC25LA012_chunk_2  |  Probable Cause and Findings  |  PIPER  | Alaska
  [5] ANC25LA013_chunk_7  |  Aircraft and Owner Info       |  CESSNA | Hawaii

ANSWER
Based on the provided NTSB accident report excerpts:
The NTSB report ANC25LA012 (Alaska) states the probable cause was the pilot's
improper airspeed management which resulted in a bounced landing and insufficient
airspeed to perform a go-around procedure. This report does not indicate an engine
failure. The report ANC25LA013 describes an engine failure, but occurred in Hawaii.
The provided context does not contain enough information about an engine failure
specifically in Alaska.
```

> Notice: the model is **honest** — it doesn't hallucinate. It clearly states what it found and what's missing.

### Negative result — honest about missing context

```
Question: What caused the helicopter crash in Hawaii in 2023?

ANSWER
The retrieved context does not contain information about a helicopter accident
in Hawaii in 2023. Try broadening your filters or check that the relevant PDFs
have been downloaded and ingested.
```

Low similarity scores and metadata mismatches are reliable signals that the answer is outside the current dataset.

---

## Technology Stack

| Layer | Tool |
|---|---|
| PDF parsing | `pdfplumber` |
| Vector database | `ChromaDB` (local, persistent) |
| Embeddings | `gemini-embedding-001` — `RETRIEVAL_DOCUMENT` at ingest, `RETRIEVAL_QUERY` at query time |
| BM25 keyword index | `rank-bm25` (BM25Okapi, built in-memory at startup) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranker | Cohere `rerank-v3.5` cross-encoder |
| Query expansion | `gemini-2.5-flash` with NTSB-aware prompt |
| Answer generation | `gemini-2.5-flash` |
| Evaluation | `ragas` 0.4.x — per-sample `ascore()` API |
| Eval LLM client | `openai.AsyncOpenAI` + `instructor` (JSON mode) → Gemini backend |
