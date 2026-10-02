#!/usr/bin/env python3
"""Phase 1 · Collection — acceptance checks for the Polymarket raw catalog.

The point of `collect_raw.py` is that nothing is silently lost, so the
collection is not finished until that claim is tested rather than assumed.
This module runs the checks that would catch a partial or corrupt walk:

  1. COVERAGE   — every planned shard is `done`; the union of windows
                  covers [id_min, id_max] with no gap and no overlap.
  2. INTEGRITY  — every stored page decompresses, parses, and yields
                  exactly `n_markets` objects, and every id in the page
                  falls inside that page's window.
  3. FIDELITY   — a random sample of markets is re-fetched live and the
                  stored object is compared field-by-field.  Volatile
                  fields (prices, volumes, `updatedAt`) are expected to
                  drift and are reported separately from structural
                  fields, which must match exactly.
  4. JOIN       — share of markets whose `event_id` resolves in `events`;
                  a low share means the event walk is incomplete, and the
                  tags/category signal is missing for those markets.
  5. RECONCILE  — the derived `markets` table row count equals the sum of
                  `raw_pages.n_markets`, so no page was archived without
                  being parsed (or vice versa).

Exit status is non-zero when any check fails, so this can gate a pipeline.

Usage:
    python -m Phase1.validate_raw [--db PATH] [--sample N]
    python -m Phase1.validate_raw --no-network   # skip check 3
"""

import argparse
import gzip
import json
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (GAMMA_BASE, accent, bold, dim, err, header, info, ok, warn)
from Phase1.collect_raw import (DB_PATH, HEADERS, WINDOW, make_pool)

# Fields that legitimately change between collection and re-fetch. Any
# other difference is a fidelity failure, not drift.
VOLATILE = {
    "updatedAt", "volume", "volumeNum", "volume24hr", "volume1wk", "volume1mo",
    "volume1yr", "volumeClob", "volume24hrClob", "volume1wkClob",
    "volume1moClob", "volume1yrClob", "volume1wkAmm", "volume1moAmm",
    "volume1yrAmm", "liquidity", "liquidityNum", "liquidityClob",
    "outcomePrices", "lastTradePrice", "bestBid", "bestAsk", "spread",
    "oneDayPriceChange", "oneHourPriceChange", "oneWeekPriceChange",
    "oneMonthPriceChange", "oneYearPriceChange", "competitive",
    "closed", "active", "archived", "closedTime", "acceptingOrders",
    "acceptingOrdersTimestamp", "events", "umaResolutionStatuses",
    "clobRewards", "rewardsMinSize", "rewardsMaxSpread", "featured", "new",
    "openInterest", "commentCount", "ready", "funded", "deploying",
}

DEFAULT_SAMPLE = 60


def _pages(conn):
    return conn.execute(
        "SELECT id_lo, id_hi, closed, n_markets, encoding, body FROM raw_pages")


def check_coverage(conn, p) -> bool:
    total, done, errored, pending = conn.execute(
        "SELECT COUNT(*), SUM(state='done'), SUM(state='error'), "
        "SUM(state='pending') FROM shards").fetchone()
    p(bold("  1. COVERAGE"))
    p(f"     shards       {total:,}   done {done or 0:,}   "
      f"error {errored or 0:,}   pending {pending or 0:,}")
    good = (errored or 0) == 0 and (pending or 0) == 0 and total > 0

    # A gap in the window grid means an id range was never asked about at
    # all — invisible in the shard states, which only describe rows that
    # exist.
    los = [r[0] for r in conn.execute(
        "SELECT DISTINCT id_lo FROM shards ORDER BY id_lo")]
    gaps = [(a, b) for a, b in zip(los, los[1:]) if b - a != WINDOW]
    if gaps:
        good = False
        p(err(f"     {len(gaps)} gap(s) in the id-window grid, e.g. "
              f"{gaps[0][0]:,} -> {gaps[0][1]:,}"))
    else:
        p(f"     grid         contiguous over {los[0]:,}..{los[-1] + WINDOW - 1:,}"
          if los else "     grid         empty")

    # Both closed passes must exist for every window: the `id` route
    # applies a default closed=false filter, so a window walked in one
    # pass only is silently missing every resolved market in it.
    lopsided = conn.execute(
        "SELECT COUNT(*) FROM (SELECT id_lo FROM shards GROUP BY id_lo "
        "HAVING COUNT(DISTINCT closed) < 2)").fetchone()[0]
    if lopsided:
        good = False
        p(err(f"     {lopsided:,} window(s) walked in only one closed pass"))
    p(ok("     PASS") if good else err("     FAIL"))
    return good


def check_integrity(conn, p) -> bool:
    bad_parse = bad_count = out_of_range = 0
    n_pages = n_rows = 0
    for id_lo, id_hi, closed, n, enc, body in _pages(conn):
        n_pages += 1
        try:
            text = gzip.decompress(body) if enc == "gzip" else body
            rows = json.loads(text.decode("utf-8"))
        except Exception:                              # noqa: BLE001
            bad_parse += 1
            continue
        if len(rows) != n:
            bad_count += 1
        n_rows += len(rows)
        for m in rows:
            if not (id_lo <= int(m["id"]) <= id_hi):
                out_of_range += 1
    p(bold("  2. INTEGRITY"))
    p(f"     pages        {n_pages:,}   objects {n_rows:,}")
    p(f"     unreadable   {bad_parse:,}   count-mismatch {bad_count:,}   "
      f"id-out-of-window {out_of_range:,}")
    good = bad_parse == bad_count == out_of_range == 0
    p(ok("     PASS") if good else err("     FAIL"))
    return good


def check_reconcile(conn, p) -> bool:
    """Reconcile every archived page's IDs to its derived rows exactly."""
    bad_pages = bad_ids = 0
    archived = derived = 0
    for page_id, id_lo, id_hi, enc, body in conn.execute(
            "SELECT page_id,id_lo,id_hi,encoding,body FROM raw_pages"):
        try:
            raw = gzip.decompress(body) if enc == "gzip" else body
            rows = json.loads(raw.decode("utf-8"))
            source_ids = [int(row["id"]) for row in rows]
        except Exception:                              # noqa: BLE001
            bad_pages += 1
            continue
        stored_ids = [
            row[0] for row in conn.execute(
                "SELECT market_id FROM markets WHERE page_id=?", (page_id,))
        ]
        archived += len(source_ids)
        derived += len(stored_ids)
        if len(source_ids) != len(set(source_ids)):
            bad_ids += 1
        if set(source_ids) != set(stored_ids):
            bad_ids += 1
            p(err(f"     page {page_id}: source/derived id set differs"))
    p(bold("  5. RECONCILE"))
    p(f"     archived objects {archived:,}   derived page rows {derived:,}   "
      f"bad pages {bad_pages:,}   bad id sets {bad_ids:,}")
    good = bad_pages == 0 and bad_ids == 0
    p(ok("     PASS") if good else err("     FAIL"))
    return good


def check_page_accounting(conn, p, legacy_empty_windows: bool = False) -> bool:
    p(bold("  6. PAGE ACCOUNTING"))
    states = {row[0] for row in conn.execute("SELECT DISTINCT state FROM shards")}
    good = states <= {"pending", "done", "error"}
    invalid = states - {"pending", "done", "error"}
    if invalid:
        p(err(f"     unknown shard states: {sorted(invalid)}"))
    missing = conn.execute(
        """SELECT COUNT(*) FROM shards s
           LEFT JOIN raw_pages p ON p.id_lo = s.id_lo AND p.closed = s.closed
           WHERE s.state = 'done' AND p.page_id IS NULL"""
    ).fetchone()[0]
    if missing and legacy_empty_windows:
        p(warn(f"     WARNING: {missing:,} done shards have no archived page"))
    elif missing:
        p(err(f"     {missing:,} done shards have no archived page"))
        good = False
    mismatch = conn.execute(
        """SELECT COUNT(*) FROM shards s JOIN raw_pages p
           ON p.id_lo=s.id_lo AND p.closed=s.closed
           WHERE p.n_markets != s.n_markets
              OR p.id_hi != s.id_lo + ?""", (WINDOW - 1,)
    ).fetchone()[0]
    dangling = conn.execute(
        "SELECT COUNT(*) FROM markets m LEFT JOIN raw_pages p ON p.page_id=m.page_id "
        "WHERE p.page_id IS NULL"
    ).fetchone()[0]
    if mismatch or dangling:
        p(err(f"     page/shard mismatches {mismatch:,}; dangling markets {dangling:,}"))
        good = False
    p(ok("     PASS") if good else err("     FAIL"))
    return good


def check_plan_bounds(conn, p) -> bool:
    p(bold("  7. PLAN BOUNDS"))
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    starts = [r[0] for r in conn.execute("SELECT DISTINCT id_lo FROM shards ORDER BY id_lo")]
    good = bool(starts) and meta.get("id_min") is not None and meta.get("id_max") is not None
    if good:
        lo, hi = int(meta["id_min"]), int(meta["id_max"])
        good = starts[0] == lo - lo % WINDOW and starts[-1] == hi - hi % WINDOW
        good = good and all(b - a == WINDOW for a, b in zip(starts, starts[1:]))
    closed = {r[0] for r in conn.execute("SELECT DISTINCT closed FROM shards")}
    good = good and closed == {0, 1}
    p(f"     windows {len(starts):,}; closed values {sorted(closed)}")
    p(ok("     PASS") if good else err("     FAIL"))
    return good


def check_event_side(conn, p) -> bool:
    have = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchone()[0]
    if not have:
        p("     SKIPPED: no events table")
        return True
    p(bold("  8. EVENT SIDE"))
    good = True
    states = conn.execute(
        "SELECT COUNT(*) FROM event_shards WHERE state NOT IN ('pending','done','error')"
    ).fetchone()[0]
    missing = conn.execute(
        """SELECT COUNT(*) FROM event_shards s LEFT JOIN raw_event_pages p
           ON p.id_lo=s.id_lo WHERE s.state='done' AND p.page_id IS NULL"""
    ).fetchone()[0]
    duplicates = conn.execute(
        "SELECT COUNT(*) FROM (SELECT event_id FROM events GROUP BY event_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    dangling = 0
    for (market_ids,) in conn.execute("SELECT market_ids FROM events"):
        try:
            ids = json.loads(market_ids or "[]")
        except (TypeError, ValueError):
            good = False
            continue
        dangling += conn.execute(
            "SELECT COUNT(*) FROM (SELECT value FROM json_each(?) "
            "EXCEPT SELECT market_id FROM markets)", (json.dumps(ids),)
        ).fetchone()[0]
    if states or missing or duplicates or dangling:
        p(err(f"     invalid states {states}; missing pages {missing}; "
              f"duplicate events {duplicates}; dangling market ids {dangling}"))
        good = False
    p(ok("     PASS") if good else err("     FAIL"))
    return good

def check_join(conn, p) -> bool:
    have_events = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='events'"
    ).fetchone()[0]
    p(bold("  4. JOIN (markets -> events)"))
    if not have_events:
        p(warn("     events table absent — run collect_events_raw first"))
        return True
    total, matched = conn.execute(
        "SELECT COUNT(*), SUM(CASE WHEN e.event_id IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM markets m LEFT JOIN events e "
        "ON e.event_id = CAST(m.event_id AS INTEGER)").fetchone()
    share = (matched or 0) / max(total, 1)
    p(f"     markets      {total:,}   with an event row {matched or 0:,} "
      f"({100 * share:.1f}%)")
    good = share >= 0.98
    p(ok("     PASS") if good else warn("     LOW — event walk likely incomplete"))
    return good


def check_fidelity(conn, p, sample: int, seed: int) -> bool:
    p(bold("  3. FIDELITY (live re-fetch)"))
    ids = [r[0] for r in conn.execute(
        "SELECT market_id FROM markets ORDER BY market_id")]
    if not ids:
        p(warn("     no markets to sample"))
        return True
    rng = random.Random(seed)
    pick = rng.sample(ids, min(sample, len(ids)))

    stored = {}
    for mid in pick:
        row = conn.execute(
            "SELECT p.body, p.encoding FROM markets m JOIN raw_pages p "
            "ON p.page_id = m.page_id WHERE m.market_id = ?", (mid,)).fetchone()
        text = gzip.decompress(row[0]) if row[1] == "gzip" else row[0]
        for m in json.loads(text.decode("utf-8")):
            if int(m["id"]) == mid:
                stored[mid] = m

    pool = make_pool(4)
    live = {}
    status_errors = 0
    for i in range(0, len(pick), WINDOW):
        chunk = pick[i:i + WINDOW]
        for closed in ("true", "false"):
            url = (f"{GAMMA_BASE}/markets?limit=500&closed={closed}&"
                   + "&".join(f"id={m}" for m in chunk))
            resp = pool.request("GET", url, headers=HEADERS, retries=False,
                                decode_content=True)
            if resp.status != 200:
                status_errors += 1
                continue
            for m in json.loads(resp.data.decode("utf-8")):
                live[int(m["id"])] = m

    missing = [m for m in pick if m not in live]
    structural_diff = []
    drifted = 0
    for mid, s in stored.items():
        l = live.get(mid)
        if l is None:
            continue
        keys = (set(s) | set(l)) - VOLATILE
        diff = [k for k in keys if s.get(k) != l.get(k)]
        if diff:
            structural_diff.append((mid, diff[:6]))
        if any(s.get(k) != l.get(k) for k in VOLATILE & (set(s) | set(l))):
            drifted += 1
    p(f"     sampled      {len(pick):,}   re-fetched {len(live):,}   "
      f"absent live {len(missing):,}")
    p(f"     status errors {status_errors:,}   missing ids {len(missing):,}")
    p(f"     volatile drift {drifted:,} (expected)   "
      f"structural mismatch {len(structural_diff):,}")
    for mid, diff in structural_diff[:5]:
        p(err(f"       market {mid}: {diff}"))
    good = not structural_diff and status_errors == 0 and not missing
    p(ok("     PASS") if good else err("     FAIL"))
    return good


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="validate_raw",
        description="Phase 1 · Collection — validate the Polymarket raw catalog.")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help="Markets to re-fetch for the fidelity check")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-network", action="store_true",
                    help="Skip the live re-fetch")
    ap.add_argument("--legacy-empty-windows", action="store_true",
                    help="Warn rather than fail for pre-archival empty windows")
    args = ap.parse_args()
    p = lambda s: print(s, flush=True)
    try:
        conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"no raw catalog schema at {args.db}: run the collector first"
        ) from exc
    try:
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"raw_pages", "shards", "markets"} <= tables:
            raise RuntimeError(
                f"no raw catalog schema at {args.db}: run the collector first"
            )
        p("")
        p(header("=== MarketGPR  ·  Polymarket Raw Catalog Validation ==="))
        p(accent(f"DB: {args.db}"))
        p("")
        results = [
            check_coverage(conn, p),
            check_integrity(conn, p),
            check_page_accounting(conn, p, args.legacy_empty_windows),
            check_plan_bounds(conn, p),
        ]
        if not args.no_network:
            results.append(check_fidelity(conn, p, args.sample, args.seed))
        results.extend([check_join(conn, p), check_event_side(conn, p),
                        check_reconcile(conn, p)])
        p("")
        if all(results):
            p(ok("  ALL CHECKS PASSED"))
            return 0
        p(err(f"  {results.count(False)} CHECK(S) FAILED"))
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
