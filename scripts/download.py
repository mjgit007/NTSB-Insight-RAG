import pandas as pd
import requests
import os
import time
import logging
import argparse
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# NOTE: The API requires the numeric Mkey field, NOT the NtsbNo string.
# NtsbNo is used only as the output filename for human readability.
BASE_URL = "https://data.ntsb.gov/carol-repgen/api/Aviation/ReportMain/GenerateNewestReport/{mkey}/pdf"

RATE_LIMIT_DELAY = 0.2      # seconds between requests within a batch (small delay to avoid 403s)
BATCH_SIZE       = 30       # number of records per batch
BATCH_DELAY      = 1.0      # seconds to pause after each batch of 30
RETRY_DELAY      = 5.0      # seconds to wait before retrying a failed request
MAX_RETRIES      = 3        # max retry attempts per record
REQUEST_TIMEOUT  = 30       # seconds before a request times out

# ---------------------------------------------------------------------------
# Argument parsing — lets you pick which CSV to run
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Download NTSB aviation accident PDFs')
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to the input CSV file (e.g. data/NTSB_2020_2024.csv)'
    )
    parser.add_argument(
        '--out-dir',
        default=None,
        help='Directory to save PDFs (default: pdfs/<csv_stem>/)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help=f'Number of records to download per batch (default: {BATCH_SIZE})'
    )
    parser.add_argument(
        '--batch-delay',
        type=float,
        default=BATCH_DELAY,
        help=f'Seconds to pause between batches (default: {BATCH_DELAY})'
    )
    parser.add_argument(
        '--only-completed',
        action='store_true',
        default=True,
        help='Skip records where ReportStatus != Completed (default: True)'
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Download a single PDF with retry logic
# ---------------------------------------------------------------------------
def download_report(ntsb_no, mkey, out_dir, delay, attempt=1):
    url = BASE_URL.format(mkey=mkey)
    pdf_path = os.path.join(out_dir, f"{ntsb_no}.pdf")  # filename uses NtsbNo for readability

    # Skip if already downloaded (resume support)
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        log.info(f"[SKIP]     {ntsb_no} (mkey={mkey}) already exists, skipping.")
        return 'skipped'

    try:
        log.debug(f"[ATTEMPT {attempt}] GET {url}")
        response = requests.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            # Sanity check: NTSB returns HTML error pages with 200 status sometimes
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and len(response.content) < 5000:
                log.warning(f"[WARN]     {ntsb_no} (mkey={mkey}) -> 200 but suspicious content ({content_type}, {len(response.content)} bytes). Skipping.")
                return 'suspicious'

            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            log.info(f"[OK]       {ntsb_no} (mkey={mkey}) -> {pdf_path} ({len(response.content)//1024} KB)")
            return 'ok'

        elif response.status_code == 404:
            log.warning(f"[404]      {ntsb_no} (mkey={mkey}) -> Report not found.")
            return 'not_found'

        elif response.status_code == 429:
            wait = RETRY_DELAY * attempt
            log.warning(f"[429]      {ntsb_no} (mkey={mkey}) -> Rate limited. Waiting {wait}s before retry...")
            time.sleep(wait)
            if attempt < MAX_RETRIES:
                return download_report(ntsb_no, mkey, out_dir, delay, attempt + 1)
            return 'rate_limited'

        elif response.status_code >= 500:
            wait = RETRY_DELAY * attempt
            log.warning(f"[{response.status_code}]      {ntsb_no} (mkey={mkey}) -> Server error. Waiting {wait}s before retry...")
            time.sleep(wait)
            if attempt < MAX_RETRIES:
                return download_report(ntsb_no, mkey, out_dir, delay, attempt + 1)
            return 'server_error'

        else:
            log.error(f"[{response.status_code}]    {ntsb_no} (mkey={mkey}) -> Unexpected status. Skipping.")
            return 'failed'

    except requests.exceptions.Timeout:
        log.warning(f"[TIMEOUT]  {ntsb_no} (mkey={mkey}) -> Request timed out (attempt {attempt}/{MAX_RETRIES}).")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)
            return download_report(ntsb_no, mkey, out_dir, delay, attempt + 1)
        return 'timeout'

    except requests.exceptions.ConnectionError as e:
        log.warning(f"[CONN_ERR] {ntsb_no} (mkey={mkey}) -> Connection error: {e} (attempt {attempt}/{MAX_RETRIES}).")
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)
            return download_report(ntsb_no, mkey, out_dir, delay, attempt + 1)
        return 'connection_error'

    except Exception as e:
        log.error(f"[ERROR]    {ntsb_no} (mkey={mkey}) -> Unexpected error: {e}")
        return 'failed'

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load CSV
    log.info(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    log.info(f"Total records in CSV: {len(df)}")

    # Filter to completed reports only (PDFs only exist for these)
    if args.only_completed and 'ReportStatus' in df.columns:
        before = len(df)
        df = df[df['ReportStatus'] == 'Completed']
        log.info(f"Filtered to Completed reports: {len(df)} (skipped {before - len(df)} In-work records)")

    # Output directory — default to docs/<csv_stem>/
    if args.out_dir:
        out_dir = args.out_dir
    else:
        csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'pdfs', csv_stem)

    os.makedirs(out_dir, exist_ok=True)
    log.info(f"PDFs will be saved to: {out_dir}")
    log.info(f"Batch size: {args.batch_size} records | Batch delay: {args.batch_delay}s between batches")
    log.info(f"Log file: {log_file}")
    log.info("-" * 60)

    # Counters
    stats = {'ok': 0, 'skipped': 0, 'not_found': 0, 'suspicious': 0,
             'rate_limited': 0, 'server_error': 0, 'timeout': 0,
             'connection_error': 0, 'failed': 0}

    # Validate Mkey column exists
    if 'Mkey' not in df.columns:
        log.error("CSV does not contain 'Mkey' column. Cannot build API URL.")
        return

    total = len(df)
    rows = list(df[['NtsbNo', 'Mkey']].itertuples(index=False))
    num_batches = (total + args.batch_size - 1) // args.batch_size

    for batch_num in range(num_batches):
        batch_start = batch_num * args.batch_size
        batch_end   = min(batch_start + args.batch_size, total)
        batch       = rows[batch_start:batch_end]

        log.info(f"--- Batch {batch_num + 1}/{num_batches} | Records {batch_start + 1}-{batch_end} of {total} ---")

        for row in batch:
            i = rows.index(row) + 1
            ntsb_no, mkey = row.NtsbNo, int(row.Mkey)
            log.info(f"[{i}/{total}] Processing {ntsb_no} (mkey={mkey})")
            result = download_report(ntsb_no, mkey, out_dir, args.batch_delay)
            stats[result] = stats.get(result, 0) + 1
            time.sleep(RATE_LIMIT_DELAY)  # small delay between requests within a batch

        # Pause between batches (not after the last one)
        if batch_num < num_batches - 1:
            log.info(f"--- Batch {batch_num + 1} complete. Pausing {args.batch_delay}s before next batch ---")
            time.sleep(args.batch_delay)

    # Summary
    log.info("=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("=" * 60)
    for status, count in stats.items():
        if count > 0:
            log.info(f"  {status:<20} : {count}")
    log.info(f"  {'TOTAL':<20} : {total}")
    log.info("=" * 60)
    log.info(f"Log saved to: {log_file}")

if __name__ == '__main__':
    main()
