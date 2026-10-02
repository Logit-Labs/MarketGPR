#!/usr/bin/env python3
"""Phase 1 · Collection — Polymarket full-fidelity raw event catalog.

Companion to `collect_raw.py`.  Markets carry an embedded `events` array,
but that embedded copy is a stub: it has no `tags` and no `category` —
the only categorical signal Polymarket publishes.  Those live exclusively
on the `/events` route, so the event catalog is collected separately and
joined back on `markets.event_id`.

Enumeration is the same id-window walk as the market collector: `/events`
accepts up to 100 repeated `id` params.  ONE pass suffices here — unlike
`/markets`, the events id route applies no default `closed` filter
(verified: a 100-id batch returns 100 events with no filter, and splits
99/1 across closed=true/false).

Storage — trimmed, then gzipped
------------------------------
Each event embeds the FULL market object of every market it contains, so
a verbatim archive would store the entire market catalog a second time:
measured 8.6 kB gzipped per event, ~8 GB for the id space, ~90% of it a
byte-for-byte duplicate of `raw_pages`.  Instead each event's `markets`
array is replaced by the list of market ids before the page is gzipped —
0.9 kB/event, ~0.9 GB — and the discarded objects remain recoverable from
`raw_pages` by that id.  This is the one place in the raw collectors
where the stored bytes are not exactly what the server sent; `raw_event_pages.trimmed`
records it, and `raw_event_pages.body` is otherwise complete.

Usage:
    python -m Phase1.collect_events_raw [--db PATH] [--workers N]
    python -m Phase1.collect_events_raw --status
"""

import argparse
import gzip
import hashlib
import json
import os
import queue
import random
import sqlite3
DISCOVERY_REQUESTS: list[dict] = []
import sys
import threading
import time
from datetime import datetime, timezone
import urllib3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (GAMMA_BASE, HttpError, accent, bold, dim, header, info, ok,
                    warn, build_url)
from Phase1.collect_raw import (DB_PATH, DEFAULT_RPS, HEADERS,
                                ID_HEADROOM, MAX_BACKOFF,
                                MAX_RETRIES, PAGE_LIMIT, SCHEMA_VERSION,
                                STATUSES, THROTTLE_PAUSE,
                                WINDOW, COMMIT_PAGES, RateLimiter,
                                Stats, ensure_page_columns,
                                init_raw_db, make_pool, _f, _s)

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_event_pages (
    page_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    id_lo      INTEGER NOT NULL,
    id_hi      INTEGER NOT NULL,
    n_events   INTEGER NOT NULL,
    trimmed    INTEGER NOT NULL DEFAULT 1,
    fetched_at TEXT NOT NULL,
    body       BLOB NOT NULL,
    payload_sha256 TEXT NOT NULL DEFAULT '',
    status INTEGER NOT NULL DEFAULT 0,
    attempt INTEGER NOT NULL DEFAULT 0,
    collector_version TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_event_pages_shard
    ON raw_event_pages(id_lo);

CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY,
    slug          TEXT,
    ticker        TEXT,
    title         TEXT,
    description   TEXT,
    category      TEXT,
    tags          TEXT,
    tag_ids       TEXT,
    series_slug   TEXT,
    created_at    TEXT,
    start_date    TEXT,
    end_date      TEXT,
    closed_time   TEXT,
    closed        INTEGER,
    active        INTEGER,
    archived      INTEGER,
    neg_risk      INTEGER,
    volume        REAL,
    open_interest REAL,
    market_ids    TEXT,
    market_count  INTEGER,
    page_id       INTEGER NOT NULL,
    fetched_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_slug ON events(slug);

CREATE TABLE IF NOT EXISTS event_shards (
    id_lo      INTEGER PRIMARY KEY,
    state      TEXT NOT NULL DEFAULT 'pending',
    n_events   INTEGER NOT NULL DEFAULT 0,
    attempts   INTEGER NOT NULL DEFAULT 0,
    error      TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_event_shards_state ON event_shards(state);
"""

EVENT_COLS = ("event_id", "slug", "ticker", "title", "description", "category",
              "tags", "tag_ids", "series_slug", "created_at", "start_date",
              "end_date", "closed_time", "closed", "active", "archived",
              "neg_risk", "volume", "open_interest", "market_ids",
              "market_count", "page_id", "fetched_at")


def fetch_window(pool, id_lo: int, delay: float,
                 limiter: "RateLimiter | None" = None) -> tuple:
    """Fetch one event window and return its provenance."""
    params = [("limit", PAGE_LIMIT)]
    params += [("id", i) for i in range(id_lo, id_lo + WINDOW)]
    url = f"{GAMMA_BASE}/events?" + "&".join(f"{k}={v}" for k, v in params)

    for attempt in range(MAX_RETRIES):
        try:
            if limiter is not None:
                limiter.acquire()
            resp = pool.request("GET", url, headers=HEADERS, retries=False,
                                decode_content=True, preload_content=True)
            STATUSES.add(resp.status)
            if resp.status in (403, 429) or resp.status >= 500:
                time.sleep(min(2 ** attempt + random.uniform(0, 1.0),
                               MAX_BACKOFF))
                continue
            if resp.status >= 400:
                raise HttpError(resp.status, url, resp.data[:400])
            decoded = resp.data
            payload_sha = hashlib.sha256(decoded).hexdigest()
            events = json.loads(decoded.decode("utf-8"))
            if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
                raise ValueError("response payload is not a list of objects")
            ids = [int(e["id"]) for e in events]
            if len(ids) != len(set(ids)):
                raise ValueError("response contains duplicate event ids")
            if any(not (id_lo <= eid < id_lo + WINDOW) for eid in ids):
                raise ValueError("response contains an id outside its requested window")
            for e in events:
                e["markets"] = [m.get("id") for m in (e.get("markets") or [])]
            body = gzip.compress(
                json.dumps(events, separators=(",", ":")).encode("utf-8"), 6)
            if delay:
                time.sleep(delay)
            return body, events, resp.status, attempt + 1, payload_sha
        except (urllib3.exceptions.HTTPError, OSError, ValueError, KeyError) as exc:
            STATUSES.add(type(exc).__name__)
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"{type(exc).__name__}: {exc}") from exc
            time.sleep(min(2 ** attempt + random.uniform(0, 1.0), MAX_BACKOFF))
    raise RuntimeError(f"exhausted {MAX_RETRIES} attempts for id_lo={id_lo}")


def discover_bounds(pool) -> tuple:
    lo, hi = None, None
    DISCOVERY_REQUESTS.clear()
    for asc in ("true", "false"):
        url = build_url("/events", {"limit": 1, "order": "id", "ascending": asc},
                        base=GAMMA_BASE)
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
        raise RuntimeError("could not determine Gamma event id bounds")
    return lo, hi


def derive(e: dict, page_id: int, fetched_at: str) -> tuple:
    """Flatten one event.  `tags` keeps the labels (the usable signal);
    `tag_ids` keeps the ids so a label rename stays traceable."""
    tags = [t for t in (e.get("tags") or []) if isinstance(t, dict)]
    series = e.get("series") or []
    market_ids = [str(m) for m in (e.get("markets") or []) if m]
    return (
        int(e["id"]), _s(e.get("slug")), _s(e.get("ticker")), _s(e.get("title")),
        _s(e.get("description")), _s(e.get("category")),
        json.dumps([t.get("label", "") for t in tags]),
        json.dumps([str(t.get("id", "")) for t in tags]),
        _s(e.get("seriesSlug") or (series[0].get("slug") if series and
                                   isinstance(series[0], dict) else "")),
        _s(e.get("createdAt")), _s(e.get("startDate")), _s(e.get("endDate")),
        _s(e.get("closedTime")),
        1 if e.get("closed") else 0,
        1 if e.get("active") else 0,
        1 if e.get("archived") else 0,
        1 if e.get("negRisk") else 0,
        _f(e.get("volume")), _f(e.get("openInterest")),
        json.dumps(market_ids), len(market_ids), page_id, fetched_at,
    )


def plan_shards(conn, id_lo: int, id_hi: int) -> int:
    rows = [(lo,) for lo in
            range(id_lo - (id_lo % WINDOW), id_hi + WINDOW, WINDOW)]
    conn.executemany(
        "INSERT OR IGNORE INTO event_shards(id_lo, state) VALUES (?, 'pending')",
        rows)
    conn.commit()
    return len(rows)


def writer_loop(conn, q: queue.Queue):
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
        id_lo, body, events, fetched_at, *extra = item
        status = extra[0] if len(extra) > 0 else 0
        attempt = extra[1] if len(extra) > 1 else 0
        payload_sha = extra[2] if len(extra) > 2 else ""
        error = extra[3] if len(extra) > 3 else None
        now = datetime.now(timezone.utc).isoformat()
        if error:
            conn.execute("UPDATE event_shards SET state='error', "
                         "attempts=attempts+1, error=?, updated_at=? WHERE id_lo=?",
                         (str(error)[:400], now, id_lo))
        else:
            cur = conn.execute(
                "INSERT OR REPLACE INTO raw_event_pages"
                "(id_lo, id_hi, n_events, trimmed, fetched_at, body,"
                " payload_sha256, status, attempt, collector_version) "
                "VALUES (?,?,?,1,?,?,?,?,?,?)",
                (id_lo, id_lo + WINDOW - 1, len(events), fetched_at, body,
                 payload_sha, status, attempt, str(SCHEMA_VERSION)))
            page_id = cur.lastrowid
            if events:
                conn.executemany(
                    "INSERT OR REPLACE INTO events(%s) VALUES (%s)"
                    % (",".join(EVENT_COLS), ",".join("?" * len(EVENT_COLS))),
                    [derive(e, page_id, fetched_at) for e in events])
            conn.execute("UPDATE event_shards SET state='done', n_events=?, "
                         "attempts=attempts+1, error='', updated_at=? WHERE id_lo=?",
                         (len(events), now, id_lo))
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
            (id_lo,) = shards.get_nowait()
        except queue.Empty:
            return
        fetched_at = datetime.now(timezone.utc).isoformat()
        try:
            body, events, status, attempt, payload_sha = fetch_window(
                pool, id_lo, delay, limiter)
            out.put((id_lo, body, events, fetched_at, status, attempt,
                     payload_sha, None))
            with stats.lock:
                stats.pages += 1
                stats.markets += len(events)
                stats.bytes += len(body)
        except Exception as exc:                       # noqa: BLE001
            out.put((id_lo, b"", [], fetched_at, 0, 0, "",
                     f"{type(exc).__name__}: {exc}"))
            with stats.lock:
                stats.pages += 1
                stats.errors += 1


def write_manifest(conn, db_path: str, *, status: str = "complete",
                    argv: list[str] | None = None,
                    started_at: str | None = None) -> str:
    n_events, n_pages, n_bytes = conn.execute(
        "SELECT (SELECT COUNT(*) FROM events), COUNT(*), "
        "COALESCE(SUM(LENGTH(body)), 0) FROM raw_event_pages").fetchone()
    states = dict(conn.execute(
        "SELECT state, COUNT(*) FROM event_shards GROUP BY state").fetchall())
    statuses = dict(conn.execute(
        "SELECT status, COUNT(*) FROM raw_event_pages GROUP BY status").fetchall())
    payload = {
        "collector": "events", "collector_version": str(SCHEMA_VERSION),
        "bounds_discovery": DISCOVERY_REQUESTS,
        "schema_version": SCHEMA_VERSION, "code_revision": None,
        "db_path": os.path.abspath(db_path), "argv": argv or sys.argv,
        "id_min": None, "id_max": None, "window": WINDOW, "rps": None,
        "workers": None, "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(), "status": status,
        "shard_states": states, "shards_total": sum(states.values()),
        "pages_total": n_pages, "pages_empty": conn.execute(
            "SELECT COUNT(*) FROM raw_event_pages WHERE n_events=0").fetchone()[0],
        "rows_total": n_events, "status_histogram": statuses,
        "bounds_user_supplied": False, "validator": None, "raw_bytes": n_bytes,
    }
    path = os.path.splitext(db_path)[0] + "_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path

def print_status(conn, p, stats: Stats | None = None) -> None:
    rows = dict(conn.execute(
        "SELECT state, COUNT(*) FROM event_shards GROUP BY state").fetchall())
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_pages, n_bytes = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(body)),0) FROM raw_event_pages").fetchone()
    p("")
    p(bold("  Event shards"))
    for state in ("done", "error", "pending"):
        if rows.get(state):
            p(f"    {state:<9} {rows[state]:>9,}")
    p(bold("  Catalog"))
    p(f"    events    {n_events:>9,}")
    p(f"    raw pages {n_pages:>9,}   {n_bytes / 1e9:.2f} GB stored")
    if stats:
        el = time.time() - stats.started
        p(f"    elapsed   {el / 60:>9.1f} min")
    if rows.get("error"):
        p(warn(f"  {rows['error']:,} shards errored — re-run to retry them."))
    p("")


def run_collection(conn, workers: int, delay: float, limit: int | None,
                   rps: float = DEFAULT_RPS) -> None:
    p = lambda s: print(s, flush=True)
    todo = conn.execute("SELECT id_lo FROM event_shards WHERE state != 'done' "
                        "ORDER BY id_lo").fetchall()
    if limit:
        todo = todo[:limit]
    if not todo:
        p(ok("Nothing pending — the event id space is fully walked."))
        return
    p(info(f"Pending event shards: {len(todo):,}"))
    p("")

    pool = make_pool(workers)
    shard_q: queue.Queue = queue.Queue()
    for s in todo:
        shard_q.put(s)
    out_q: queue.Queue = queue.Queue(maxsize=workers * 8)
    stats = Stats()
    stop = threading.Event()
    limiter = RateLimiter(rps)

    wr = threading.Thread(target=writer_loop, args=(conn, out_q), daemon=True)
    wr.start()
    threads = [threading.Thread(target=worker_loop,
                                args=(pool, shard_q, out_q, stats, stop, delay,
                                      limiter),
                                daemon=True) for _ in range(workers)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(2.0)
            with stats.lock:
                done, ev, by, er = stats.pages, stats.markets, stats.bytes, stats.errors
            el = time.time() - stats.started
            rate = done / max(el, 1e-9)
            p(dim(f"  {done:,}/{len(todo):,} pages  {ev:,} events  "
                  f"{by / 1e6:,.0f} MB  {rate:.1f} pg/s  "
                  f"ETA {(len(todo) - done) / max(rate, 1e-9) / 60:.1f} min  "
                  f"errors {er:,}  [{STATUSES.summary()}]"))
    except KeyboardInterrupt:
        p(warn("\nInterrupted — draining writes and checkpointing shards."))
        stop.set()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=30)
        out_q.put(None)
        wr.join(timeout=60)
        conn.commit()
    print_status(conn, p, stats)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="collect_events_raw",
        description="Phase 1 · Collection — Polymarket raw event catalog.")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--rps", type=float, default=DEFAULT_RPS,
                    help=f"Process-wide request ceiling (default {DEFAULT_RPS})")
    ap.add_argument("--allow-unpaced", action="store_true",
                    help="Explicitly allow --rps 0")
    ap.add_argument("--id-min", type=int, default=None)
    ap.add_argument("--id-max", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.rps == 0 and not args.allow_unpaced:
        ap.error("--rps 0 requires --allow-unpaced")
    p = lambda s: print(s, flush=True)

    conn = init_raw_db(args.db)
    conn.executescript(SCHEMA)
    ensure_page_columns(conn, "raw_event_pages")
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    if args.status:
        print_status(conn, p)
        conn.close()
        return
    started_at = datetime.now(timezone.utc).isoformat()
    write_manifest(conn, args.db, status="running",
                   argv=sys.argv, started_at=started_at)

    pool = make_pool(4)
    if args.id_min is None or args.id_max is None:
        d_lo, d_hi = discover_bounds(pool)
    else:
        d_lo = d_hi = 0
    id_lo = args.id_min if args.id_min is not None else d_lo
    id_hi = (args.id_max if args.id_max is not None else d_hi) + ID_HEADROOM

    p("")
    p(header("=== MarketGPR  ·  Polymarket Raw Event Catalog ==="))
    p(accent(f"DB:       {args.db}"))
    p(accent(f"Id range: {id_lo:,} .. {id_hi:,}  ({WINDOW} ids/request, 1 pass)"))
    p(accent(f"Workers:  {args.workers}   ceiling {args.rps or 0:g} req/s"))
    p("")
    n = plan_shards(conn, id_lo, id_hi)
    p(info(f"Planned {n:,} event shards."))
    if args.dry_run:
        print_status(conn, p)
        p(dim("  dry-run — nothing fetched."))
        return
    run_collection(conn, args.workers, args.delay, args.limit, args.rps)
    manifest = write_manifest(conn, args.db, status="complete",
                              argv=sys.argv, started_at=started_at)
    p(dim(f"Manifest: {manifest}"))
    conn.close()


if __name__ == "__main__":
    main()
