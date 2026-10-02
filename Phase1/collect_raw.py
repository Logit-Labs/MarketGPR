#!/usr/bin/env python3
"""Phase 1 · Collection — Polymarket full-fidelity raw catalog.

Supersedes `collect_catalog.py` for archival collection.  That script walks
the `/markets/keyset` cursor chain and keeps 17 of the ~90 fields Gamma
ships; everything else is discarded at collection time and is unrecoverable
if the API ever changes or goes away.  This module instead stores the
**verbatim HTTP response body** for every market and derives typed columns
from it, so any field can be re-parsed later without re-fetching.

Enumeration strategy — id windows, not cursors
----------------------------------------------
`GET /markets` accepts the `id` query parameter repeated up to 100 times,
and returns the identical market object the keyset route returns (verified
field-for-field on the same market).  So the catalog is enumerated by
walking the integer id space in windows of 100:

    GET /markets?limit=500&closed=<c>&id=N&id=N+1&...&id=N+99

That makes the walk embarrassingly parallel (no cursor chain to serialize),
exactly resumable (each window is a row in `shards`), and verifiable (the
union of windows provably covers [id_min, id_max] with no gaps).  It also
sidesteps the two failure modes of the keyset walk: Gamma's WAF 403s any
request carrying both a date filter and an `after_cursor`, and pre-2023
markets carry a null `startDate` so date sharding cannot see them at all.

TWO PASSES ARE REQUIRED.  The `id` route applies Gamma's default
`closed=false` filter, so ids of resolved markets come back empty unless
`closed=true` is sent explicitly — verified: ids 12,17,18,19,20 return 0
rows by default and 5 rows with `closed=true`.  `closed` is a boolean on
every market, so {closed=true, closed=false} partitions the catalog.

Storage — gzip on the wire IS the archive
-----------------------------------------
Requests send `Accept-Encoding: gzip` and the gzip bytes are stored
unmodified in `raw_pages.body` (`decode_content=False`).  Measured on live
pages: 6.9 kB/market uncompressed, 0.55 kB/market gzipped — a 10x cut to
both bandwidth and disk, at zero CPU cost, and the blob is a byte-exact
copy of what the server sent.  Compressing each market separately instead
would cost 2.6 kB/market (~4.8x more) because a 7 kB JSON document is too
small to amortize a zlib dictionary.

Projected full run (~2.9M markets, id space 12..~4.05M):
    ~81k requests · ~1.7 GB on disk · ~1.9 GB over the wire

Usage:
    python -m Phase1.collect_raw [--db PATH] [--workers N]
    python -m Phase1.collect_raw --dry-run      # plan only
    python -m Phase1.collect_raw --status       # progress of a run

Re-running resumes: only shards not already marked `done` are fetched.
"""

import argparse
import gzip
import hashlib
import json
import os
import queue
import random
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone

import urllib3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (DATA_DIR, GAMMA_BASE, HttpError, accent, bold, dim, err,
                    header, highlight, info, ok, warn, USER_AGENT, build_url)

WINDOW = 100            # Gamma rejects >100 repeated `id` params with HTTP 422
PAGE_LIMIT = 500        # `limit` must exceed WINDOW or the page is truncated
COMMIT_PAGES = 200      # pages buffered per write transaction
MAX_RETRIES = 8
MAX_BACKOFF = 30.0
SCHEMA_VERSION = 2
ID_HEADROOM = 5_000     # ids minted between bound discovery and the walk
DISCOVERY_REQUESTS: list[dict] = []
DEFAULT_RPS = 12.0      # sustained request ceiling; see RateLimiter
THROTTLE_PAUSE = 1.0    # 429 backoff; the bucket refills in ~1s, see below

SOURCE_DIR = os.path.join(DATA_DIR, "Phase1", "Polymarket")
os.makedirs(SOURCE_DIR, exist_ok=True)
DB_PATH = os.path.join(SOURCE_DIR, "polymarket_raw.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_pages (
    page_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    id_lo      INTEGER NOT NULL,
    id_hi      INTEGER NOT NULL,
    closed     INTEGER NOT NULL,
    n_markets  INTEGER NOT NULL,
    encoding   TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    body       BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL DEFAULT '',
    status INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    collector_version TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_pages_shard ON raw_pages(id_lo, closed);

CREATE TABLE IF NOT EXISTS markets (
    market_id        INTEGER PRIMARY KEY,
    condition_id     TEXT,
    question_id      TEXT,
    slug             TEXT,
    question         TEXT,
    description      TEXT,
    event_id         TEXT,
    event_ids        TEXT,
    event_slug       TEXT,
    event_title      TEXT,
    created_at       TEXT,
    start_date       TEXT,
    end_date         TEXT,
    closed_time      TEXT,
    closed           INTEGER,
    active           INTEGER,
    archived         INTEGER,
    outcomes         TEXT,
    outcome_prices   TEXT,
    clob_token_ids   TEXT,
    last_trade_price REAL,
    volume_num       REAL,
    liquidity_num    REAL,
    neg_risk         INTEGER,
    page_id          INTEGER NOT NULL,
    fetched_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_markets_condition ON markets(condition_id);
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_id);
CREATE INDEX IF NOT EXISTS idx_markets_created ON markets(created_at);

CREATE TABLE IF NOT EXISTS shards (
    id_lo      INTEGER NOT NULL,
    closed     INTEGER NOT NULL,
    state      TEXT NOT NULL DEFAULT 'pending',
    n_markets  INTEGER NOT NULL DEFAULT 0,
    attempts   INTEGER NOT NULL DEFAULT 0,
    error      TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (id_lo, closed)
);
CREATE INDEX IF NOT EXISTS idx_shards_state ON shards(state);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

MARKET_COLS = ("market_id", "condition_id", "question_id", "slug", "question",
               "description", "event_id", "event_ids", "event_slug",
               "event_title", "created_at", "start_date", "end_date",
               "closed_time", "closed", "active", "archived", "outcomes",
               "outcome_prices", "clob_token_ids", "last_trade_price",
               "volume_num", "liquidity_num", "neg_risk", "page_id",
               "fetched_at")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class RateLimiter:
    """Process-wide token bucket over all fetch threads.

    Measured against `/markets` 100-id windows (500 requests per tier):

        target  achieved   p50     statuses
          4       4.0     0.25s    200:200
          8       8.0     0.25s    200:300
         16      15.7     0.26s    200:400
         32      31.6     0.25s    200:481  429:19
        none     81.9     0.21s    200:253  429:247

    So the sustained ceiling is ~16 req/s; past it Gamma sheds load, and
    at full speed it rejects half of everything sent.  An unthrottled
    16-worker collector run measured 28 pages/s for ~1,200 requests and
    then collapsed to 0.6 pages/s and stayed there — every worker sitting
    in exponential backoff on 429s, with `errors 0` in the shard table,
    so nothing in the run said why.  Backing off AFTER the fact is
    strictly worse than not exceeding the ceiling.  Pace the whole
    process, not each worker."""

    def __init__(self, rps: float):
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = time.monotonic()

    def acquire(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if wait:
            time.sleep(wait)


class Statuses:
    """Response-status histogram, so a throttled run says so."""

    def __init__(self):
        self.lock = threading.Lock()
        self.counts: dict = {}

    def add(self, code) -> None:
        with self.lock:
            self.counts[code] = self.counts.get(code, 0) + 1

    def summary(self) -> str:
        with self.lock:
            items = sorted(self.counts.items(), key=lambda kv: str(kv[0]))
        return " ".join(f"{k}:{v}" for k, v in items) or "-"


STATUSES = Statuses()

def make_pool(workers: int) -> urllib3.PoolManager:
    """One pooled connection per worker.

    `common._HTTP_POOL` is deliberately sized maxsize=1 for the
    single-threaded enrichment; sharing it across N fetch threads would
    serialize every request behind one socket."""
    # block=True: with block=False urllib3 discards connections past
    # maxsize and re-handshakes TLS for them, which is pure loss when the
    # worker count is the concurrency ceiling anyway.
    return urllib3.PoolManager(
        num_pools=2, maxsize=workers, retries=False, block=True,
        timeout=urllib3.Timeout(connect=10, read=120),
    )


HEADERS = {
    "Accept": "application/json",
    # Gamma honours gzip and ships ~10x less; the encoded bytes ARE the
    # archive, so this is not merely a transfer optimization.
    "Accept-Encoding": "gzip",
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}


def fetch_window(pool, id_lo: int, closed: bool, delay: float,
                 limiter: "RateLimiter | None" = None) -> tuple:
    """Fetch one id window and return body provenance plus parsed markets."""
    params = [("limit", PAGE_LIMIT), ("closed", "true" if closed else "false")]
    params += [("id", i) for i in range(id_lo, id_lo + WINDOW)]
    url = f"{GAMMA_BASE}/markets?" + "&".join(f"{k}={v}" for k, v in params)

    for attempt in range(MAX_RETRIES):
        try:
            if limiter is not None:
                limiter.acquire()
            resp = pool.request("GET", url, headers=HEADERS, retries=False,
                                decode_content=False, preload_content=True)
            STATUSES.add(resp.status)
            if resp.status in (403, 429) or resp.status >= 500:
                if resp.status == 429:
                    time.sleep(THROTTLE_PAUSE + random.uniform(0, 0.5))
                else:
                    time.sleep(min(2 ** attempt + random.uniform(0, 1.0),
                                   MAX_BACKOFF))
                continue
            if resp.status >= 400:
                raise HttpError(resp.status, url, resp.data[:400])
            body = resp.data
            encoding = (resp.headers.get("Content-Encoding") or "identity").lower()
            decoded = gzip.decompress(body) if encoding == "gzip" else body
            payload_sha = hashlib.sha256(decoded).hexdigest()
            payload = json.loads(decoded.decode("utf-8"))
            if not isinstance(payload, list) or not all(isinstance(m, dict) for m in payload):
                raise ValueError("response payload is not a list of objects")
            ids = [int(m["id"]) for m in payload]
            if len(ids) != len(set(ids)):
                raise ValueError("response contains duplicate market ids")
            if any(not (id_lo <= mid < id_lo + WINDOW) for mid in ids):
                raise ValueError("response contains an id outside its requested window")
            if delay:
                time.sleep(delay)
            return body, encoding, payload, resp.status, attempt + 1, payload_sha
        except (urllib3.exceptions.HTTPError, OSError, ValueError, KeyError) as exc:
            STATUSES.add(type(exc).__name__)
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(min(2 ** attempt + random.uniform(0, 1.0), MAX_BACKOFF))
    raise RuntimeError(f"exhausted {MAX_RETRIES} attempts for id_lo={id_lo}")


def discover_bounds(pool) -> tuple:
    """Return (min_id, max_id) over both closed values with status logging."""
    lo, hi = None, None
    DISCOVERY_REQUESTS.clear()
    for closed in ("true", "false"):
        for asc in ("true", "false"):
            url = build_url("/markets",
                            {"limit": 1, "order": "id", "ascending": asc,
                             "closed": closed}, base=GAMMA_BASE)
            last_status = None
            for attempt in range(MAX_RETRIES):
                at = datetime.now(timezone.utc).isoformat()
                resp = pool.request("GET", url, headers=HEADERS, retries=False,
                                    decode_content=True)
                last_status = resp.status
                DISCOVERY_REQUESTS.append(
                    {"url": url, "status": resp.status, "attempt": attempt + 1,
                     "at": at})
                if resp.status == 200:
                    break
                if resp.status == 429 or resp.status >= 500:
                    time.sleep(min(2 ** attempt + random.uniform(0, 1), MAX_BACKOFF))
                    continue
                raise RuntimeError(f"bounds discovery HTTP {resp.status}: {url}")
            else:
                raise RuntimeError(
                    f"bounds discovery exhausted after {MAX_RETRIES} attempts "
                    f"(last status {last_status})")
            rows = json.loads(resp.data.decode("utf-8"))
            if not rows:
                continue
            v = int(rows[0]["id"])
            lo = v if lo is None else min(lo, v)
            hi = v if hi is None else max(hi, v)
    if lo is None:
        raise RuntimeError("could not determine Gamma id bounds")
    return lo, hi


# --------------------------------------------------------------------------
# DB
# --------------------------------------------------------------------------

def ensure_page_columns(conn: sqlite3.Connection, table: str) -> None:
    columns = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})")
    }
    additions = (
        ("payload_sha256", "TEXT NOT NULL DEFAULT ''"),
        ("status", "INTEGER NOT NULL DEFAULT 0"),
        ("attempt", "INTEGER NOT NULL DEFAULT 0"),
        ("collector_version", "TEXT NOT NULL DEFAULT ''"),
    )
    for name, definition in additions:
        if name not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def init_raw_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=120, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=120000")
    found = conn.execute("PRAGMA user_version").fetchone()[0]
    if found > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(
            f"raw catalog schema v{found} is newer than this collector "
            f"(v{SCHEMA_VERSION}); refusing to write"
        )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-131072")
    conn.executescript(SCHEMA)
    ensure_page_columns(conn, "raw_pages")
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    return conn


def _s(v):
    return "" if v is None else str(v)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def derive(m: dict, page_id: int, fetched_at: str) -> tuple:
    """Flatten one market into the typed query columns.

    Deliberately lossy — `raw_pages.body` holds the complete object, so
    this table exists only to make joins and filters fast.  `created_at`
    is carried because it is present on EVERY market, while `start_date`
    is null on the pre-2023 catalog."""
    events = m.get("events") or []
    ev = events[0] if events else {}
    return (
        int(m["id"]),
        _s(m.get("conditionId")), _s(m.get("questionID")), _s(m.get("slug")),
        _s(m.get("question")), _s(m.get("description")),
        _s(ev.get("id")),
        json.dumps([str(e.get("id")) for e in events if e.get("id")]),
        _s(ev.get("slug")), _s(ev.get("title")),
        _s(m.get("createdAt")), _s(m.get("startDate")), _s(m.get("endDate")),
        _s(ev.get("closedTime") or m.get("closedTime")),
        1 if m.get("closed") else 0,
        1 if m.get("active") else 0,
        1 if m.get("archived") else 0,
        _s(m.get("outcomes")), _s(m.get("outcomePrices")),
        _s(m.get("clobTokenIds")),
        _f(m.get("lastTradePrice")), _f(m.get("volumeNum")),
        _f(m.get("liquidityNum")),
        1 if m.get("negRisk") else 0,
        page_id, fetched_at,
    )


def plan_shards(conn, id_lo: int, id_hi: int) -> int:
    """Materialize the full window list.  Idempotent: existing rows (and
    their `done` state) survive, so widening the id range on a later run
    only adds the new tail."""
    rows = [(lo, c)
            for lo in range(id_lo - (id_lo % WINDOW), id_hi + WINDOW, WINDOW)
            for c in (1, 0)]
    conn.executemany(
        "INSERT OR IGNORE INTO shards(id_lo, closed, state) VALUES (?,?,'pending')",
        rows)
    conn.commit()
    return len(rows)


def pending_shards(conn) -> list:
    """Pending shards, ordered so a window's two passes run back to back.

    ORDERING IS CORRECTNESS, NOT TASTE.  Walking pass-by-pass
    (`ORDER BY closed DESC, id_lo`) leaves ~2 hours between a window's
    closed=true fetch and its closed=false fetch, and a market that
    settles in that interval is returned by NEITHER: it was still open
    during the closed=true pass and is already closed by the closed=false
    pass.  Measured on the first full run — 183 markets lost, every one of
    them with a `closedTime` inside the run window.  Interleaving the two
    passes per window shrinks that race from the length of the run to the
    round-trip of one request."""
    return conn.execute(
        "SELECT id_lo, closed FROM shards WHERE state != 'done' "
        "ORDER BY id_lo, closed DESC").fetchall()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.pages = 0
        self.markets = 0
        self.bytes = 0
        self.errors = 0
        self.started = time.time()


def writer_loop(conn, q: queue.Queue, stats: Stats, stop: threading.Event):
    """Single-threaded DB writer that archives every successful page."""
    buffered = 0
    last_commit = time.time()
    while True:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            if buffered:
                conn.commit()
                buffered = 0
            continue
        if item is None:
            break
        id_lo, closed, body, encoding, markets, fetched_at, *extra = item
        status = extra[0] if len(extra) > 0 else 0
        attempt = extra[1] if len(extra) > 1 else 0
        payload_sha = extra[2] if len(extra) > 2 else ""
        error = extra[3] if len(extra) > 3 else None
        now = datetime.now(timezone.utc).isoformat()
        if error:
            conn.execute(
                "UPDATE shards SET state='error', attempts=attempts+1, "
                "error=?, updated_at=? WHERE id_lo=? AND closed=?",
                (str(error)[:400], now, id_lo, closed))
        else:
            cur = conn.execute(
                "INSERT OR REPLACE INTO raw_pages"
                "(id_lo, id_hi, closed, n_markets, encoding, fetched_at, body,"
                " payload_sha256, status, attempt, collector_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (id_lo, id_lo + WINDOW - 1, closed, len(markets), encoding,
                 fetched_at, body, payload_sha, status, attempt, str(SCHEMA_VERSION)))
            page_id = cur.lastrowid
            if markets:
                conn.executemany(
                    "INSERT OR REPLACE INTO markets(%s) VALUES (%s)"
                    % (",".join(MARKET_COLS), ",".join("?" * len(MARKET_COLS))),
                    [derive(m, page_id, fetched_at) for m in markets])
            conn.execute(
                "UPDATE shards SET state='done', n_markets=?, attempts=attempts+1, "
                "error='', updated_at=? WHERE id_lo=? AND closed=?",
                (len(markets), now, id_lo, closed))
        buffered += 1
        q.task_done()
        if buffered >= COMMIT_PAGES or time.time() - last_commit > 10:
            conn.commit()
            buffered = 0
            last_commit = time.time()
    conn.commit()


def worker_loop(pool, shards: queue.Queue, out: queue.Queue, stats: Stats,
                stop: threading.Event, delay: float,
                limiter: "RateLimiter | None" = None):
    while not stop.is_set():
        try:
            id_lo, closed = shards.get_nowait()
        except queue.Empty:
            return
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            body, encoding, markets, status, attempt, payload_sha = fetch_window(
                pool, id_lo, bool(closed), delay, limiter)
            out.put((id_lo, closed, body, encoding, markets, fetched_at,
                     status, attempt, payload_sha, None))
            with stats.lock:
                stats.pages += 1
                stats.markets += len(markets)
                stats.bytes += len(body)
        except Exception as exc:                       # noqa: BLE001
            out.put((id_lo, closed, b"", "", [], fetched_at, 0, 0, "",
                     f"{type(exc).__name__}: {exc}"))
            with stats.lock:
                stats.pages += 1
                stats.errors += 1


def run_collection(conn, workers: int, delay: float, limit: int | None,
                   rps: float = DEFAULT_RPS) -> None:
    p = lambda s: print(s, flush=True)
    todo = pending_shards(conn)
    if limit:
        todo = todo[:limit]
    if not todo:
        p(ok("Nothing pending — the id space is fully walked."))
        return

    total = conn.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
    p(info(f"Pending shards: {len(todo):,} of {total:,}"))
    p("")

    pool = make_pool(workers)
    shard_q: queue.Queue = queue.Queue()
    for s in todo:
        shard_q.put(s)
    out_q: queue.Queue = queue.Queue(maxsize=workers * 8)
    stats = Stats()
    stop = threading.Event()
    limiter = RateLimiter(rps)

    wr = threading.Thread(target=writer_loop, args=(conn, out_q, stats, stop),
                          daemon=True)
    wr.start()
    threads = [threading.Thread(target=worker_loop,
                                args=(pool, shard_q, out_q, stats, stop, delay,
                                      limiter),
                                daemon=True)
               for _ in range(workers)]
    for t in threads:
        t.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(2.0)
            with stats.lock:
                done, mk, by, er = stats.pages, stats.markets, stats.bytes, stats.errors
            el = time.time() - stats.started
            rate = done / max(el, 1e-9)
            eta = (len(todo) - done) / max(rate, 1e-9)
            p(dim(f"  {done:,}/{len(todo):,} pages  {mk:,} markets  "
                  f"{by / 1e6:,.0f} MB  {rate:.1f} pg/s  "
                  f"ETA {eta / 60:.1f} min  errors {er:,}  "
                  f"[{STATUSES.summary()}]"))
    except KeyboardInterrupt:
        p(warn("\nInterrupted — draining writes and checkpointing shards."))
        stop.set()
        for t in threads:
            t.join(timeout=10)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
        out_q.put(None)
        wr.join(timeout=60)
        conn.commit()

    print_status(conn, p, stats)


def write_manifest(conn, db_path: str, *, status: str = "complete",
                    argv: list[str] | None = None,
                    started_at: str | None = None,
                    bounds_user_supplied: bool = False) -> str:
    n_markets, n_pages, n_bytes = conn.execute(
        "SELECT (SELECT COUNT(*) FROM markets), COUNT(*), "
        "COALESCE(SUM(LENGTH(body)), 0) FROM raw_pages").fetchone()
    states = dict(conn.execute(
        "SELECT state, COUNT(*) FROM shards GROUP BY state").fetchall())
    statuses = dict(conn.execute(
        "SELECT status, COUNT(*) FROM raw_pages GROUP BY status").fetchall())
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    payload = {
        "collector": "markets",
        "collector_version": str(SCHEMA_VERSION),
        "bounds_discovery": DISCOVERY_REQUESTS,
        "schema_version": SCHEMA_VERSION,
        "code_revision": None,
        "db_path": os.path.abspath(db_path),
        "argv": argv or sys.argv,
        "api_base": GAMMA_BASE,
        "route": "/markets?id=... (100 ids/request, closed=true + closed=false)",
        "id_min": meta.get("id_min"), "id_max": meta.get("id_max"),
        "window": WINDOW, "rps": None, "workers": None,
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": status, "shard_states": states, "shards_total": sum(states.values()),
        "pages_total": n_pages, "pages_empty": conn.execute(
            "SELECT COUNT(*) FROM raw_pages WHERE n_markets=0").fetchone()[0],
        "rows_total": n_markets, "status_histogram": statuses,
        "bounds_user_supplied": bounds_user_supplied, "validator": None,
        "markets": n_markets, "raw_pages": n_pages, "raw_bytes": n_bytes,
        "complete": states.get("done", 0) == sum(states.values()),
    }
    path = os.path.splitext(db_path)[0] + "_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def print_status(conn, p, stats: Stats | None = None) -> None:
    rows = dict(conn.execute(
        "SELECT state, COUNT(*) FROM shards GROUP BY state").fetchall())
    n_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    n_pages, n_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(body)),0) FROM raw_pages").fetchone()
    p("")
    p(bold("  Shards"))
    for state in ("done", "error", "pending"):
        if rows.get(state):
            p(f"    {state:<9} {rows[state]:>9,}")
    p(bold("  Catalog"))
    p(f"    markets   {n_markets:>9,}")
    p(f"    raw pages {n_pages:>9,}   {n_bytes / 1e9:.2f} GB stored")
    if stats:
        el = time.time() - stats.started
        p(f"    elapsed   {el / 60:>9.1f} min   {stats.bytes / 1e6 / max(el, 1e-9):.1f} MB/s")
    if rows.get("error"):
        p("")
        p(warn(f"  {rows['error']:,} shards errored — re-run to retry them "
               f"(only non-done shards are refetched)."))
        for id_lo, closed, e in conn.execute(
                "SELECT id_lo, closed, error FROM shards WHERE state='error' LIMIT 5"):
            p(dim(f"    id={id_lo} closed={closed}: {e[:120]}"))
    p("")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="collect_raw",
        description="Phase 1 · Collection — Polymarket raw catalog (full fidelity).")
    ap.add_argument("--db", default=DB_PATH, help="Output SQLite database")
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent fetch threads (default 12)")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Per-worker pause after each request (seconds)")
    ap.add_argument("--rps", type=float, default=DEFAULT_RPS,
                    help=f"Process-wide request ceiling (default {DEFAULT_RPS}); "
                         f"0 disables pacing")
    ap.add_argument("--allow-unpaced", action="store_true",
                    help="Explicitly allow --rps 0")
    ap.add_argument("--id-min", type=int, default=None,
                    help="Override the discovered id floor")
    ap.add_argument("--id-max", type=int, default=None,
                    help="Override the discovered id ceiling")
    ap.add_argument("--limit", type=int, default=None,
                    help="Fetch at most N shards this run (smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover bounds and plan shards; fetch nothing")
    ap.add_argument("--status", action="store_true",
                    help="Print progress of an existing run and exit")
    ap.add_argument("--resweep", action="store_true",
                    help="Reset every shard to pending and walk again, "
                         "catching markets that settled mid-run (see "
                         "`pending_shards`). INSERT OR REPLACE dedups.")
    ap.add_argument("--resweep-ids", default=None, metavar="LO,HI",
                    help="Reset only the windows covering this id range")
    args = ap.parse_args()
    if args.rps == 0 and not args.allow_unpaced:
        ap.error("--rps 0 requires --allow-unpaced")
    p = lambda s: print(s, flush=True)

    conn = init_raw_db(args.db)
    if args.status:
        print_status(conn, p)
        conn.close()
        return
    started_at = datetime.now(timezone.utc).isoformat()
    write_manifest(conn, args.db, status="running",
                   argv=sys.argv, started_at=started_at,
                   bounds_user_supplied=args.id_min is not None or args.id_max is not None)

    if args.resweep or args.resweep_ids:
        if args.resweep_ids:
            lo, hi = (int(x) for x in args.resweep_ids.split(","))
            n = conn.execute("UPDATE shards SET state='pending' "
                             "WHERE id_lo BETWEEN ? AND ?",
                             (lo - lo % WINDOW, hi)).rowcount
        else:
            n = conn.execute("UPDATE shards SET state='pending'").rowcount
        conn.commit()
        p(info(f"Resweep: {n:,} shards reset to pending."))

    pool = make_pool(4)
    if args.id_min is None or args.id_max is None:
        d_lo, d_hi = discover_bounds(pool)
    else:
        d_lo = d_hi = 0
    id_lo = args.id_min if args.id_min is not None else d_lo
    id_hi = (args.id_max if args.id_max is not None else d_hi) + ID_HEADROOM

    p("")
    p(header("=== MarketGPR  ·  Polymarket Raw Catalog ==="))
    p(accent(f"API:      {GAMMA_BASE}"))
    p(accent(f"DB:       {args.db}"))
    p(accent(f"Id range: {id_lo:,} .. {id_hi:,}  ({WINDOW} ids/request, 2 passes)"))
    p(accent(f"Workers:  {args.workers}   ceiling {args.rps or 0:g} req/s"))
    p("")

    n = plan_shards(conn, id_lo, id_hi)
    conn.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                     [("id_min", str(id_lo)), ("id_max", str(id_hi)),
                      ("window", str(WINDOW)),
                      ("planned_at", datetime.now(timezone.utc).isoformat())])
    conn.commit()
    p(info(f"Planned {n:,} shards ({n // 2:,} id windows x 2 closed passes)."))

    if args.dry_run:
        print_status(conn, p)
        p(dim("  dry-run — nothing fetched."))
        return

    run_collection(conn, args.workers, args.delay, args.limit, args.rps)
    manifest = write_manifest(
        conn, args.db, status="complete", argv=sys.argv,
        started_at=started_at,
        bounds_user_supplied=args.id_min is not None or args.id_max is not None)
    p(dim(f"Manifest: {manifest}"))
    conn.close()


if __name__ == "__main__":
    main()
