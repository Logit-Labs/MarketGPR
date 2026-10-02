#!/usr/bin/env python3
"""Shared helpers for the MarketGPR phase scripts.

Provides:
    * Canonical contracts table schema (single source of truth)
    * Connection management (writable, read-only with REGEXP)
    * Collection manifest generation
    * API URL building and retrying fetch (GET + POST)
    * ANSI colour constants (dark-blue palette)
"""

import hashlib
import json
import os
import random
import re
import sqlite3
import time
import urllib.parse
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional
import urllib3

# Shared keep-alive connection for _http_json — a fresh TLS handshake per
# request dominates the per-call cost in bulk enrichment (2.8M+ calls).
# Single-threaded callers: one pooled connection per process is enough.
_HTTP_POOL = urllib3.PoolManager(
    num_pools=1,
    maxsize=1,
    retries=False,
    timeout=urllib3.Timeout(connect=10, read=60),
)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"
MAX_RETRIES = 8
BACKOFF_BASE = 2
MAX_BACKOFF = 30.0
# 403 = Cloudflare-style edge block (HTML challenge page, not an API
# denial).  urllib cannot execute the JS challenge, so a short backoff
# ladder just hammers a block that lifts only when the CF window expires.
# The block persists longer than 3.5 min (observed), and the caller
# (collect_catalog.collect_pass) commits and saves a checkpoint before
# the process exits, so a long in-process wait is safe.
BLOCK_RETRIES = 4
BLOCK_SLEEPS = (60.0, 300.0, 600.0, 1200.0)  # 36 min total; observed CF window > 3.5 min

# Gamma API sits behind Cloudflare and returns 403 Error 1010 for the
# default urllib UA; a browser-like UA passes.
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/126.0.0.0 Safari/537.36")

# Repo-local data directory; override with MARKETGPR_DATA_DIR.
DATA_DIR = os.environ.get(
    "MARKETGPR_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
)
os.makedirs(DATA_DIR, exist_ok=True)

CREATE_CONTRACTS = """CREATE TABLE IF NOT EXISTS contracts (
    ticker        TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    sub_title     TEXT NOT NULL DEFAULT '',
    event_ticker  TEXT NOT NULL DEFAULT '',
    expiry_date   TEXT NOT NULL,
    fetched_at    TEXT NOT NULL,
    market_data   TEXT NOT NULL DEFAULT '',
    price_history TEXT NOT NULL DEFAULT '',
    ohlc          TEXT NOT NULL DEFAULT '',
    result        TEXT NOT NULL DEFAULT '',
    settlement_ts TEXT NOT NULL DEFAULT '',
    volume        TEXT NOT NULL DEFAULT '',
    open_time     TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT '',
    title         TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '',
    last_price    TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    slug          TEXT NOT NULL DEFAULT ''
)"""

CREATE_EXPIRY_IDX  = "CREATE INDEX IF NOT EXISTS idx_expiry ON contracts(expiry_date)"

class HttpError(RuntimeError):
    """Non-retryable HTTP status from _http_json (4xx except 403/429).
    Carries the status code and raw body so callers can branch without
    parsing the message; subclasses RuntimeError so generic handlers
    (checkpoint + re-raise, chunk skip) keep working."""

    def __init__(self, status: int, url: str, body: bytes):
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} for {url} ({body[:200]!r})")



BOLD  = "\033[1m"
DIM   = "\033[2m"
RESET = "\033[0m"

BLUE        = "\033[34m"
BRIGHT_BLUE = "\033[94m"
CYAN        = "\033[36m"
GREEN       = "\033[32m"
YELLOW      = "\033[33m"
RED         = "\033[31m"
MAGENTA     = "\033[35m"


def header(text: str) -> str:
    return f"{BOLD}{BRIGHT_BLUE}{text}{RESET}"


def info(text: str) -> str:
    return f"{CYAN}{text}{RESET}"


def ok(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def warn(text: str) -> str:
    return f"{YELLOW}{text}{RESET}"


def err(text: str) -> str:
    return f"{RED}{text}{RESET}"


def accent(text: str) -> str:
    return f"{BLUE}{text}{RESET}"


def dim(text: str) -> str:
    return f"{DIM}{text}{RESET}"


def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def highlight(text: str) -> str:
    return f"{BOLD}{BRIGHT_BLUE}{text}{RESET}"


def init_db(db_path: str) -> sqlite3.Connection:
    """Create / open a writable SQLite connection with performance PRAGMAs.
    Migrates old schemas by ALTERing in any columns missing from CREATE_CONTRACTS."""
    # check_same_thread=False: collect runs concurrent fetch workers that
    # share this connection, fully serialized by a caller-side lock.
    conn = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
    # Wait (up to 60s) instead of failing when a parallel worker holds the
    # write lock — required for concurrent enrich workers on one DB.
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute(CREATE_CONTRACTS)
    conn.execute(CREATE_EXPIRY_IDX)

    cols = {r[1] for r in conn.execute("PRAGMA table_info(contracts)").fetchall()}
    if "event_ticker" not in cols:
        conn.execute(
            "ALTER TABLE contracts ADD COLUMN event_ticker TEXT NOT NULL DEFAULT ''"
        )
    if "sub_title" not in cols:
        conn.execute(
            "ALTER TABLE contracts ADD COLUMN sub_title TEXT NOT NULL DEFAULT ''"
        )
    if "market_data" not in cols:
        conn.execute(
            "ALTER TABLE contracts ADD COLUMN market_data TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            "ALTER TABLE contracts ADD COLUMN price_history TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(
            "ALTER TABLE contracts ADD COLUMN ohlc TEXT NOT NULL DEFAULT ''"
        )
    # Unified cross-source columns; old DBs get them via ALTER.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(contracts)").fetchall()}
    for col in ("result", "settlement_ts", "volume", "open_time", "status",
                "title", "category", "tags", "last_price", "description",
                "slug"):
        if col not in cols:
            conn.execute(
                f"ALTER TABLE contracts ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
            )

    conn.commit()
    return conn


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """Open a read-only SQLite connection.  REGEXP is registered on it."""
    abs_path = os.path.abspath(db_path)
    uri = f"file:{abs_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    _register_regex(conn)
    return conn


def _register_regex(conn: sqlite3.Connection) -> None:
    """Register a case-insensitive regexp() user function."""

    def _regexp(pattern: str, text: Optional[str]) -> bool:
        if text is None:
            return False
        try:
            return re.search(pattern, text, re.IGNORECASE) is not None
        except re.error:
            return False

    conn.create_function("REGEXP", 2, _regexp, deterministic=True)


def write_manifest(db_path: str, start_ts: int, end_ts: int,
                   duration: float, api_base: str,
                   manifest_name: str = "collection_manifest.json") -> str:
    """Write a manifest JSON into DATA_DIR.  Return path."""
    sha = hashlib.sha256()
    with open(db_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)

    conn = sqlite3.connect(db_path)
    row_count = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    conn.close()

    manifest = {
        "api_base":         api_base,
        "start_ts":         start_ts,
        "end_ts":           end_ts,
        "start_iso":        datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
        "end_iso":          datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        "total_rows":       row_count,
        "duration_seconds": round(duration, 1),
        "db_sha256":        sha.hexdigest(),
        "finished_at":      datetime.now(timezone.utc).isoformat(),
    }

    manifest_path = os.path.join(DATA_DIR, manifest_name)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def build_url(path: str, params: dict | None = None,
              base: str = GAMMA_BASE) -> str:
    """Build a full API URL with query string.  `params` values that are
    lists become repeated query keys (doseq)."""
    url = f"{base}{path}"
    if params:
        cleaned = {k: v for k, v in params.items() if v is not None}
        if cleaned:
            qs = urllib.parse.urlencode(cleaned, doseq=True)
            url = f"{url}?{qs}"
    return url


def _http_json(url: str, data: bytes | None = None) -> dict:
    """HTTP GET (data=None) or POST (data=body bytes) with retry/backoff,
    over a pooled keep-alive connection (avoids a fresh TLS handshake per
    call — the dominant per-request cost in the bulk enrichment).
    Returns parsed JSON dict."""
    # Full browser-like header set: Cloudflare bot-scoring weights header
    # presence/order, and intermittently challenged the lean urllib profile
    # (HTTP 403 challenge pages) during the bulk enrichment tail.
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-CH-UA": '"Chromium";v="126", "Google Chrome";v="126", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"macOS"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    method = "POST" if data is not None else "GET"
    for attempt in range(MAX_RETRIES):
        try:
            resp = _HTTP_POOL.request(method, url, body=data, headers=headers,
                                      retries=False)
            if resp.status == 403:
                if attempt >= BLOCK_RETRIES:
                    raise RuntimeError(
                        f"Edge block persists after {BLOCK_RETRIES} long waits "
                        f"({url}); CF window likely >{BLOCK_SLEEPS[-1]:.0f}s (36 min waited) — "
                        f"wait and re-run (checkpoint resumes the walk)")
                wait = BLOCK_SLEEPS[min(attempt, len(BLOCK_SLEEPS) - 1)]
                print(warn(f"HTTP 403 edge block (attempt {attempt + 1}/{BLOCK_RETRIES}): "
                           f"{resp.data[:200]!r} — waiting {wait:.0f}s"), flush=True)
                time.sleep(wait)
                continue
            if resp.status == 429 or resp.status >= 500:
                if resp.status == 429:
                    # Token-bucket throttle: the bucket refills in ~0.25s, so a
                    # short fixed pause keeps attempts near the cap.  The
                    # exponential ladder (2^attempt, up to 30s) starved
                    # attempts below the bucket under sustained load.
                    wait = 1.0 + random.uniform(0, 0.5)
                else:
                    wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 1.0),
                               MAX_BACKOFF)
                print(warn(f"HTTP {resp.status} (attempt {attempt + 1}/{MAX_RETRIES}): "
                           f"{resp.data[:200]!r} — retrying in {wait:.1f}s"), flush=True)
                time.sleep(wait)
                continue
            if resp.status >= 400:
                raise HttpError(resp.status, url, resp.data)
            return json.loads(resp.data.decode("utf-8"))
        except (urllib3.exceptions.HTTPError, OSError) as exc:
            wait = min(BACKOFF_BASE ** attempt + random.uniform(0, 1.0),
                       MAX_BACKOFF)
            print(warn(f"{type(exc).__name__} (attempt {attempt + 1}/{MAX_RETRIES}): "
                       f"{exc} — retrying in {wait:.1f}s"), flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")


def fetch_json(url: str) -> dict:
    """HTTP GET with retry/backoff.  Returns parsed JSON dict."""
    return _http_json(url)


def post_json(url: str, payload: dict | list) -> dict:
    """HTTP POST JSON payload (dict or list) with retry/backoff.
    Returns parsed JSON dict."""
    return _http_json(url, json.dumps(payload).encode("utf-8"))
