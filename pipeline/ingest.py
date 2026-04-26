"""
ingest.py - NTSB PDF ingestion pipeline

Stages:
  1. Load CSV metadata (keyed by NtsbNo)
  2. Extract text from PDF using pdfplumber
  3. Split into section-aware chunks
  4. Attach CSV metadata to each chunk
  5. Write chunks to a JSONL file for next stage (embed.py)

Usage:
  python ingest.py --pdf-dir ../docs/NTSB_2020_2024 \
                   --csv     ../data/NTSB_2020_2024.csv \
                   --out     chunks_2020_2024.jsonl
"""

import os
import re
import json
import argparse
import logging
from datetime import datetime

import pandas as pd
import pdfplumber

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
# Section headers found in NTSB reports (ALL CAPS patterns)
# Order matters — we match top-down
# ---------------------------------------------------------------------------
SECTION_PATTERNS = [
    (re.compile(r'Analysis', re.IGNORECASE),                        'Analysis'),
    (re.compile(r'Probable Cause and Findings', re.IGNORECASE),     'Probable Cause and Findings'),
    (re.compile(r'Findings', re.IGNORECASE),                        'Findings'),
    (re.compile(r'Factual Information', re.IGNORECASE),             'Factual Information'),
    (re.compile(r'History of (the )?Flight', re.IGNORECASE),        'History of Flight'),
    (re.compile(r'Pilot Information', re.IGNORECASE),               'Pilot Information'),
    (re.compile(r'Co-?pilot Information', re.IGNORECASE),           'Copilot Information'),
    (re.compile(r'Aircraft and Owner', re.IGNORECASE),              'Aircraft and Owner Information'),
    (re.compile(r'Meteorological Information', re.IGNORECASE),      'Meteorological Information'),
    (re.compile(r'Wreckage and Impact Information', re.IGNORECASE), 'Wreckage and Impact Information'),
    (re.compile(r'Administrative Information', re.IGNORECASE),      'Administrative Information'),
    (re.compile(r'Flight Recorder', re.IGNORECASE),                 'Flight Recorder Information'),
    (re.compile(r'Airport Information', re.IGNORECASE),             'Airport Information'),
]

# Sections we don't want to embed — boilerplate / admin
SKIP_SECTIONS = {'Administrative Information'}

# Chunk token limits (approx: 1 token ~ 4 chars)
MAX_CHUNK_CHARS  = 4000   # ~1000 tokens — safe for text-embedding-004 (2048 token limit)
OVERLAP_CHARS    = 400    # ~10% overlap


# ---------------------------------------------------------------------------
# Stage 1: Load CSV metadata
# ---------------------------------------------------------------------------
def load_csv_metadata(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    # Keep only completed reports
    df = df[df['ReportStatus'] == 'Completed']
    # Replace NaN with None for clean JSON
    df = df.where(pd.notna(df), other=None)
    metadata = df.set_index('NtsbNo').to_dict('index')
    log.info(f"Loaded metadata for {len(metadata)} completed records from {csv_path}")
    return metadata


# ---------------------------------------------------------------------------
# Stage 2: Extract full text from PDF, page by page
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Strip page footers like "Page 3 of 8 ANC24FA029"
                    text = re.sub(r'Page \d+ of \d+\s*\w*\n?', '', text)
                    pages.append(text.strip())
    except Exception as e:
        log.error(f"Failed to extract text from {pdf_path}: {e}")
        return ''
    return '\n'.join(pages)


# ---------------------------------------------------------------------------
# Stage 3: Split text into sections
# Returns list of (section_name, section_text)
# ---------------------------------------------------------------------------
def split_into_sections(full_text: str) -> list[tuple[str, str]]:
    lines = full_text.split('\n')
    sections = []
    current_section = 'Header'
    current_lines   = []

    for line in lines:
        stripped = line.strip()
        matched_section = None

        # Check if this line is a section header
        for pattern, label in SECTION_PATTERNS:
            if pattern.fullmatch(stripped) or (len(stripped) < 60 and pattern.search(stripped)):
                matched_section = label
                break

        if matched_section:
            # Save previous section
            if current_lines:
                sections.append((current_section, '\n'.join(current_lines).strip()))
            current_section = matched_section
            current_lines   = []
        else:
            current_lines.append(line)

    # Save final section
    if current_lines:
        sections.append((current_section, '\n'.join(current_lines).strip()))

    # Drop empty sections and skipped ones
    sections = [
        (name, text) for name, text in sections
        if text and name not in SKIP_SECTIONS and len(text.strip()) > 50
    ]
    return sections


# ---------------------------------------------------------------------------
# Stage 3b: If a section text is too long, split with overlap
# ---------------------------------------------------------------------------
def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at a sentence boundary
        if end < len(text):
            boundary = text.rfind('. ', start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap  # overlap with previous chunk

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Stage 4: Build chunk dicts with metadata attached
# ---------------------------------------------------------------------------
def build_chunks(ntsb_no: str, sections: list[tuple[str, str]], csv_meta: dict) -> list[dict]:
    meta = csv_meta.get(ntsb_no, {})
    chunks = []
    chunk_index = 0

    for section_name, section_text in sections:
        sub_chunks = chunk_text(section_text)
        for sub_text in sub_chunks:
            chunks.append({
                'id'      : f"{ntsb_no}_chunk_{chunk_index}",
                'ntsb_no' : ntsb_no,
                'text'    : sub_text,
                'metadata': {
                    # Identity
                    'ntsb_no'        : ntsb_no,
                    'mkey'           : meta.get('Mkey'),
                    'section'        : section_name,
                    'chunk_index'    : chunk_index,
                    # Event
                    'event_date'     : str(meta.get('EventDate', '')),
                    'event_type'     : meta.get('EventType'),
                    'city'           : meta.get('City'),
                    'state'          : meta.get('State'),
                    'country'        : meta.get('Country'),
                    'latitude'       : meta.get('Latitude'),
                    'longitude'      : meta.get('Longitude'),
                    'airport_id'     : meta.get('AirportID'),
                    'airport_name'   : meta.get('AirportName'),
                    # Aircraft
                    'make'           : meta.get('Make'),
                    'model'          : meta.get('Model'),
                    'aircraft_cat'   : meta.get('AirCraftCategory'),
                    'aircraft_damage': meta.get('AirCraftDamage'),
                    'amateur_built'  : meta.get('AmateurBuilt'),
                    'num_engines'    : meta.get('NumberOfEngines'),
                    # Injuries
                    'injury_severity': meta.get('HighestInjuryLevel'),
                    'fatal_count'    : meta.get('FatalInjuryCount'),
                    'serious_count'  : meta.get('SeriousInjuryCount'),
                    'minor_count'    : meta.get('MinorInjuryCount'),
                    # Operation
                    'far'            : meta.get('FAR'),
                    'purpose'        : meta.get('PurposeOfFlight'),
                    'operator'       : meta.get('Operator'),
                    'weather'        : meta.get('WeatherCondition'),
                    'probable_cause' : meta.get('ProbableCause'),
                    'report_status'  : meta.get('ReportStatus'),
                }
            })
            chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='NTSB PDF ingestion — extract, chunk, enrich')
    parser.add_argument('--pdf-dir', required=True, help='Directory containing downloaded PDFs')
    parser.add_argument('--csv',     required=True, help='CSV file with NTSB metadata')
    parser.add_argument('--out',     required=True, help='Output JSONL file for chunks')
    parser.add_argument('--limit',   type=int, default=None, help='Process only N PDFs (for testing)')
    args = parser.parse_args()

    # Load metadata
    csv_meta = load_csv_metadata(args.csv)

    # Collect PDFs
    pdf_files = sorted([
        f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')
    ])
    if args.limit:
        pdf_files = pdf_files[:args.limit]

    log.info(f"Found {len(pdf_files)} PDFs in {args.pdf_dir}")

    # Stats
    stats = {'processed': 0, 'no_meta': 0, 'no_text': 0, 'no_sections': 0, 'total_chunks': 0}

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    with open(out_path, 'w') as out_file:
        for i, pdf_name in enumerate(pdf_files, start=1):
            ntsb_no  = pdf_name.replace('.pdf', '')
            pdf_path = os.path.join(args.pdf_dir, pdf_name)

            log.info(f"[{i}/{len(pdf_files)}] {ntsb_no}")

            # Check metadata exists
            if ntsb_no not in csv_meta:
                log.warning(f"  No CSV metadata for {ntsb_no}, skipping.")
                stats['no_meta'] += 1
                continue

            # Extract text
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text:
                log.warning(f"  No text extracted from {pdf_name}, skipping.")
                stats['no_text'] += 1
                continue

            # Split into sections
            sections = split_into_sections(full_text)
            if not sections:
                log.warning(f"  No sections detected in {pdf_name}, skipping.")
                stats['no_sections'] += 1
                continue

            log.info(f"  Sections: {[s for s, _ in sections]}")

            # Build chunks with metadata
            chunks = build_chunks(ntsb_no, sections, csv_meta)
            log.info(f"  Chunks: {len(chunks)}")

            # Write to JSONL
            for chunk in chunks:
                out_file.write(json.dumps(chunk) + '\n')

            stats['processed']    += 1
            stats['total_chunks'] += len(chunks)

    # Summary
    log.info("=" * 60)
    log.info("INGESTION SUMMARY")
    log.info("=" * 60)
    for k, v in stats.items():
        log.info(f"  {k:<20} : {v}")
    log.info(f"  Output file         : {out_path}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
