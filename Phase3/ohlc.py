#!/usr/bin/env python3
"""MarketGPR Phase 3 · OHLC — shared runner for the per-exchange OHLC pulls.

Copies a Phase 2 contracts DB (enriched catalog or any level DB) into
data/Phase3/<exchange>/ (default) and fills the `price_history` and `ohlc`
JSON columns contract by contract from the exchange's OHLC API.  The output
DB keeps the exact table set and schema of the input (contracts + series
label tables), so it looks like the Phase 2 DBs — only price_history / ohlc
/ ohlc_fetched_at are written (fetched_at stays the collection timestamp).
Orchestrated full runs (Phase3/orchestrate.py) write data/Phase3/<exchange>/ by passing --out-dir.

Daily-bar convention: `date` is the US/Eastern trading day the bar covers
(00:00-24:00 ET) and `t` is the period end in UTC — 00:00 ET of the
following day (04:00Z summer / 05:00Z winter).  Polymarket points are
bucketed onto this grid.

Resumable and idempotent:
  * contracts with a non-empty ohlc column are skipped unless --refresh;
  * a contract the API has no data for is stored as '[]' and counted empty;
  * a contract whose fetch failed keeps ohlc='' and is retried next run.

Parallel workers: each worker owns a disjoint rowid range
(--rowid-start/--rowid-end) and writes into the same output DB — WAL mode
plus busy_timeout=60000 absorb write contention.  A worker that dies is
re-launched on the same range; unpopulated rows are picked up by resume
semantics.
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import DATA_DIR, accent, dim, header, ok, warn


# --- US/Eastern time helpers (stdlib-only: no tzdata on this box) ----------
# US DST rule (2007+): DST from 02:00 ET on the second Sunday of March until
# 02:00 ET on the first Sunday of November.  ET = UTC-4 during DST, UTC-5
# otherwise.  Daily bars are labeled by the ET trading day (00:00-24:00 ET).

def _dst_instants(year: int) -> tuple[int, int]:
    """UTC instants when US Eastern DST starts/ends in `year`."""

    def nth_sunday(y: int, m: int, nth: int) -> datetime:
        first = datetime(y, m, 1, tzinfo=timezone.utc)
        offset = (6 - first.weekday()) % 7          # days to first Sunday
        return first + timedelta(days=offset + 7 * (nth - 1))

    start = nth_sunday(year, 3, 2) + timedelta(hours=7)   # 02:00 EST = 07:00 UTC
    end = nth_sunday(year, 11, 1) + timedelta(hours=6)   # 02:00 EDT = 06:00 UTC
    return int(start.timestamp()), int(end.timestamp())


def _eastern_offset_hours(ts: int) -> int:
    """US Eastern UTC offset in hours at `ts` (4 during DST, 5 otherwise)."""
    year = datetime.fromtimestamp(ts, tz=timezone.utc).year
    start, end = _dst_instants(year)
    return 4 if start <= ts < end else 5


def eastern_date(ts: int) -> str:
    """UTC timestamp -> US/Eastern calendar date (YYYY-MM-DD)."""
    return datetime.fromtimestamp(ts - _eastern_offset_hours(ts) * 3600,
                                  tz=timezone.utc).strftime("%Y-%m-%d")


def eastern_day_end_ts(date_str: str) -> int:
    """UTC timestamp of 00:00 ET on the day AFTER `date_str` — the end of
    that ET trading day."""
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    start_utc = int(datetime(d.year, d.month, d.day,
                             tzinfo=timezone.utc).timestamp())
    return start_utc + _eastern_offset_hours(start_utc) * 3600


def default_out_dir(exchange: str) -> str:
    return os.path.join(DATA_DIR, "Phase3", exchange)


def register_common_args(parser: argparse.ArgumentParser, exchange: str) -> None:
    parser.add_argument("--db", required=True,
                        help="Path to a Phase 2 contracts SQLite database "
                             "(catalog or level DB)")
    parser.add_argument("--out-dir", default=None,
                        help=f"Output directory (default: "
                             f"{default_out_dir(exchange)})")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N contracts (smoke test)")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch contracts that already have ohlc")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Pause between API requests (seconds)")
    parser.add_argument("--chunk", type=int, default=100,
                        help="Contracts per fetch batch")
    parser.add_argument("--rowid-start", type=int, default=None,
                        help="Only process contracts with rowid >= this "
                             "(parallel workers own disjoint rowid ranges)")
    parser.add_argument("--rowid-end", type=int, default=None,
                        help="Only process contracts with rowid <= this")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip writing ohlc_manifest.json (parallel "
                             "workers: the orchestrator writes the "
                             "aggregate manifest from the output DB)")


def open_output_db(input_db: str, out_dir: str) -> tuple[sqlite3.Connection, str]:
    """Copy the input DB into the output data dir (once) and open it."""
    os.makedirs(out_dir, exist_ok=True)
    out_db = os.path.join(out_dir, os.path.basename(input_db))
    if not os.path.isfile(out_db):
        shutil.copyfile(input_db, out_db)
    conn = sqlite3.connect(out_db, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Per-row provenance for the OHLC pull; fetched_at stays the collection
    # timestamp.  Additive ALTER — the same migration pattern as init_db.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(contracts)").fetchall()}
    if "ohlc_fetched_at" not in cols:
        conn.execute(
            "ALTER TABLE contracts ADD COLUMN ohlc_fetched_at TEXT NOT NULL DEFAULT ''"
        )
    conn.commit()
    return conn, out_db


def select_targets(conn: sqlite3.Connection, limit: int, refresh: bool,
                   rowid_start: int | None = None,
                   rowid_end: int | None = None):
    conds: list[str] = []
    if not refresh:
        conds.append("ohlc IS NULL OR ohlc = ''")
    if rowid_start is not None:
        conds.append(f"rowid >= {int(rowid_start)}")
    if rowid_end is not None:
        conds.append(f"rowid <= {int(rowid_end)}")
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    sql = ("SELECT rowid, ticker, open_time, expiry_date, market_data "
           f"FROM contracts {where} ORDER BY rowid")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def write_manifest(exchange: str, input_db: str, out_db: str, args,
                   counts: dict, duration: float) -> str:
    manifest = {
        "exchange": exchange,
        "input_db": os.path.basename(input_db),
        "output_db": os.path.basename(out_db),
        "period_minutes": getattr(args, "period", None),
        "history_start": getattr(args, "history_start", None),
        **counts,
        "duration_seconds": round(duration, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(os.path.dirname(out_db), "ohlc_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def run_collect(exchange: str, args: argparse.Namespace, fetch_batch) -> None:
    """Shared collection loop.  fetch_batch(rows, args) -> {ticker: (price_history, ohlc)}.

    A ticker absent from the returned dict is a failed fetch (ohlc stays '');
    a ticker mapped to ([], []) is a completed no-data contract ('[]')."""
    p = lambda s: print(s, flush=True)

    if not os.path.isfile(args.db):
        p(warn(f"Error: database not found: {args.db}"))
        sys.exit(1)

    out_dir = args.out_dir or default_out_dir(exchange)
    conn, out_db = open_output_db(args.db, out_dir)

    total_in_db = conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    rows = select_targets(conn, args.limit, args.refresh,
                          getattr(args, "rowid_start", None),
                          getattr(args, "rowid_end", None))
    total = len(rows)
    skipped = max(total_in_db - total, 0)

    p("")
    p(header(f"=== MarketGPR  ·  Phase 3  ·  OHLC ({exchange}) ==="))
    p(accent(f"Input:  {args.db}"))
    p(accent(f"Output: {out_db}"))
    p(accent(f"Targets: {total:,}  |  already populated: {skipped:,}"))

    started = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    done = fetched = empty = failed = 0
    next_print = 500
    for i in range(0, total, args.chunk):
        chunk = rows[i:i + args.chunk]
        try:
            results = fetch_batch(chunk, args)
        except RuntimeError as exc:
            p(warn(f"  Chunk {i // args.chunk + 1} failed: {str(exc)[:160]} — "
                   f"leaving {len(chunk)} rows for a later pass"))
            failed += len(chunk)
            continue
        for row in chunk:
            res = results.get(row["ticker"])
            if res is None:
                failed += 1
                continue
            ph, ohlc = res
            conn.execute(
                "UPDATE contracts SET price_history=?, ohlc=?, ohlc_fetched_at=? "
                "WHERE rowid=?",
                (json.dumps(ph), json.dumps(ohlc), now_iso, row["rowid"]),
            )
            done += 1
            if ohlc:
                fetched += 1
            else:
                empty += 1
        conn.commit()
        if done >= next_print:
            p(dim(f"  {done:,}/{total:,}  |  fetched={fetched:,}  empty={empty:,}  "
                  f"failed={failed:,}  ({time.time() - started:.0f}s)"))
            next_print += 500

    duration = time.time() - started
    p("")
    p(ok(f"Done: {done:,} written  |  {fetched:,} with OHLC  |  {empty:,} no data  |  "
         f"{failed:,} failed  |  {duration:.1f}s"))
    p(ok(f"Wrote: {out_db}"))
    if not getattr(args, "no_manifest", False):
        manifest = write_manifest(exchange, args.db, out_db, args,
                                  {"contracts_total": total_in_db,
                                   "written": done, "with_ohlc": fetched,
                                   "empty": empty, "failed": failed},
                                  duration)
        p(dim(f"Manifest: {manifest}"))
    conn.close()
    if failed and not args.limit:
        p(warn(f"{failed:,} contracts left unpopulated — re-run to retry "
               f"(resumes where it stopped)"))
