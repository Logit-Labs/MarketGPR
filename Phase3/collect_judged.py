#!/usr/bin/env python3
"""Phase 3b: daily price history for the judged geopolitical roster.

The existing `collect_ohlc.py` walks a Phase 2 level-1 database and its
`contracts` table. This pipeline never built one -- Phase 2 was rewritten as
an event filter over the Phase 1 raw catalog -- so this script does the same
job against the roster that `Phase2/judge_events.py` produced.

Scale is the reason it is worth a separate script rather than a schema
adapter. The raw catalog holds 2.87M markets. The judged roster is on the
order of 1,500 events and perhaps 8,000 markets, which is a few hours of
polite requests instead of a multi-day run.

What the CLOB gives us, and what it does not:

  {t, p} only. There is no per-period volume and no open interest in any
  Polymarket endpoint. Volume exists once per market, as a lifetime total,
  and the `volume1wk`/`1mo`/`1yr` fields are frozen at market close rather
  than trailing -- so no daily activity measure exists for this venue at all.
  Any weighting this project applies has to come from price, from the count
  of live markets, or from a lifetime figure that is not knowable on day t.

  The grid is forward-filled. The endpoint returns a point for every bucket
  whether or not anybody traded, so the number of points in a day carries no
  information about activity. Derived "activity" proxies measured earlier --
  distinct prices per day, daily range -- correlate negatively with volume.
  They measure price uncertainty, not liquidity. Do not revive them.

Bars are bucketed by the US/Eastern day, matching `Phase3/ohlc.py`, so a bar
labelled 2025-06-23 covers the same 24 hours as every other daily bar in this
project.

Usage:
    python -m Phase3.collect_judged --dry-run
    python -m Phase3.collect_judged
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from common import bold, dim, err, header, info, ok, warn  # noqa: E402
from Phase1.enrich_contracts import history_for_chunk  # noqa: E402
from Phase3.ohlc import eastern_date, eastern_day_end_ts  # noqa: E402

REPO = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
SRC_CANDIDATES = (os.path.join(REPO, "data", "Phase1", "Polymarket", "polymarket_raw.db"),
                  os.path.join(REPO, "data", "Phase1", "Polymarket", "events_slim.db"))
JUDGE_DB = os.path.join(REPO, "data", "Phase2", "event_judge.db")
OUT_DB = os.path.join(REPO, "data", "Phase3", "Polymarket", "judged_ohlc.db")


def _has_events_table(path: str) -> bool:
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            return bool(con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone())
    except sqlite3.Error:
        return False


def resolve_source() -> str:
    for candidate in SRC_CANDIDATES:
        if os.path.isfile(candidate) and os.path.getsize(candidate) and _has_events_table(candidate):
            return candidate
    raise FileNotFoundError("no Phase 1 catalog with an events table: " +
                            ", ".join(SRC_CANDIDATES))


SRC_DB = resolve_source()

CHUNK = 20        # markets per batch-history request group
FIDELITY = 10     # interval=1m forces fidelity >= 10 (10-minute buckets)


def load_roster(con_out, only_new: bool = True) -> list[dict]:
    """Markets belonging to events the judge marked geopolitical.

    An event can carry many markets -- `Next Country US Strikes` has one per
    candidate country -- and each is a separate CLOB series. All of them come
    down. Which ones the index uses is a decision for the index, not for the
    collector, and re-fetching later costs another full run.
    """
    j = sqlite3.connect(f"file:{JUDGE_DB}?mode=ro", uri=True)
    keep = {str(r[0]) for r in j.execute(
        "SELECT event_id FROM judgments WHERE relevant = 1")}
    j.close()
    if not keep:
        sys.exit(err("no geopolitical events in the judge database"))

    done = set()
    if only_new:
        done = {r[0] for r in con_out.execute(
            "SELECT market_id FROM markets_done")}

    src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    out = []
    for (mid, cid, eid, slug, question, start, end, closed, toks, outcomes
         ) in src.execute(
            "SELECT market_id, condition_id, event_id, slug, question, "
            "start_date, end_date, closed_time, clob_token_ids, outcomes FROM markets"):
        if str(eid) not in keep or str(mid) in done:
            continue
        token = _yes_token(toks, outcomes)
        if not token:
            try:
                labels = [str(x).strip().lower() for x in json.loads(outcomes or "[]")]
            except (ValueError, TypeError):
                labels = []
            status = "ambiguous_outcomes" if set(labels) != {"yes", "no"} or len(labels) != 2 else "no_token"
            con_out.execute(
                "INSERT OR REPLACE INTO market_status(market_id,event_id,status,reason) "
                "VALUES (?,?,?,?)", (int(mid), str(eid), status, "missing or invalid YES token"))
            continue
        out.append({"market_id": str(mid), "condition_id": cid,
                    "event_id": str(eid), "question": question,
                    "start": start, "end": closed or end, "token": token})
    con_out.commit()
    src.close()
    return out

def _yes_token(raw, outcomes=None) -> str | None:
    """Return the YES token only for an unambiguous binary market."""
    try:
        vals = json.loads(raw) if raw else []
        labels = json.loads(outcomes) if isinstance(outcomes, str) and outcomes else outcomes
    except (ValueError, TypeError):
        return None
    if not isinstance(vals, list) or len(vals) != 2:
        return None
    if labels is None:
        return None
    if (not isinstance(labels, list) or len(labels) != 2 or
            {str(x).strip().lower() for x in labels} != {"yes", "no"}):
        return None
    yes_i = [str(x).strip().lower() for x in labels].index("yes")
    return str(vals[yes_i])


def init_out() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)
    con = sqlite3.connect(OUT_DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS bars (
            market_id TEXT NOT NULL,
            event_id  TEXT NOT NULL,
            date      TEXT NOT NULL,
            t         INTEGER NOT NULL,
            open      REAL, high REAL, low REAL, close REAL,
            n_points  INTEGER,
            PRIMARY KEY (market_id, date)
        );
        CREATE TABLE IF NOT EXISTS markets_done (
            market_id TEXT PRIMARY KEY,
            event_id  TEXT,
            n_bars    INTEGER,
            at        TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS market_status (
            market_id TEXT PRIMARY KEY,
            event_id TEXT,
            status TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            n_bars INTEGER NOT NULL DEFAULT 0,
            token TEXT NOT NULL DEFAULT '',
            start_ts INTEGER, end_ts INTEGER,
            at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS ix_bars_date ON bars(date);
        CREATE INDEX IF NOT EXISTS ix_bars_event ON bars(event_id);
        CREATE INDEX IF NOT EXISTS ix_market_status ON market_status(status);
    """)
    con.commit()
    return con


def daily_bars(points: list[dict]) -> list[dict]:
    """Collapse {t, p} points into one bar per US/Eastern day."""
    by_day: dict[str, list] = {}
    for pt in points:
        ts = int(pt["t"])
        by_day.setdefault(eastern_date(ts), []).append(float(pt["p"]))
    bars = []
    for day, prices in sorted(by_day.items()):
        bars.append({"date": day, "t": eastern_day_end_ts(day),
                     "open": prices[0], "high": max(prices),
                     "low": min(prices), "close": prices[-1],
                     "n_points": len(prices)})
    return bars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--delay", type=float, default=0.35,
                    help="seconds between batch requests")
    ap.add_argument("--fidelity", type=int, default=FIDELITY)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refetch", action="store_true",
                    help="ignore markets_done and pull everything again")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the work, make no request")
    ap.add_argument("--errors-out", default=None)
    args = ap.parse_args()

    print(info(f"catalog: {SRC_DB}"))
    con = init_out()
    rows = load_roster(con, only_new=not args.refetch)
    if args.limit:
        rows = rows[:args.limit]
    n_events = len({r["event_id"] for r in rows})
    print(info(f"{len(rows):,} markets across {n_events:,} judged events"))
    if args.dry_run:
        print(warn("DRY RUN: no requests made"))
        for r in rows[:5]:
            print(dim(f"  {r['market_id']}  {str(r['question'])[:70]}"))
        con.close()
        return
    if not rows:
        print(ok("nothing to collect"))
        con.close()
        return

    now_ts = int(time.time())
    n_bars = n_empty = 0
    errors = []
    for i in range(0, len(rows), args.chunk):
        batch = rows[i:i + args.chunk]
        recs = [({"conditionId": r["condition_id"], "startDate": r["start"],
                  "endDate": r["end"]}, r["token"]) for r in batch]
        got = None
        last_exc = None
        for attempt in range(3):
            try:
                got = history_for_chunk(recs, None, now_ts, args.delay,
                                        fidelity=args.fidelity)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(min(2 ** attempt, 4))
        if got is None:
            reason = str(last_exc)
            print(warn(f"  chunk {i // args.chunk}: {reason}"))
            for r in batch:
                con.execute(
                    "INSERT OR REPLACE INTO market_status"
                    "(market_id,event_id,status,reason,token,start_ts,end_ts) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (int(r["market_id"]), r["event_id"], "failed", reason[:500],
                     r["token"], r["start"], r["end"]))
                errors.append({"market_id": r["market_id"], "event_id": r["event_id"],
                                "reason": reason})
            con.commit()
            continue
        for r in batch:
            pts = got.get(r["condition_id"]) or []
            bars = daily_bars(pts)
            if not bars:
                n_empty += 1
                status = "no_history"
            else:
                status = "done"
            for b in bars:
                con.execute(
                    "INSERT OR REPLACE INTO bars (market_id, event_id, date, "
                    "t, open, high, low, close, n_points) VALUES (?,?,?,?,?,?,?,?,?)",
                    (r["market_id"], r["event_id"], b["date"], b["t"],
                     b["open"], b["high"], b["low"], b["close"], b["n_points"]))
            con.execute("INSERT OR REPLACE INTO markets_done "
                        "(market_id, event_id, n_bars) VALUES (?,?,?)",
                        (r["market_id"], r["event_id"], len(bars)))
            con.execute(
                "INSERT OR REPLACE INTO market_status"
                "(market_id,event_id,status,n_bars,token,start_ts,end_ts) "
                "VALUES (?,?,?,?,?,?,?)",
                (int(r["market_id"]), r["event_id"], status, len(bars),
                 r["token"], r["start"], r["end"]))
            n_bars += len(bars)
        con.commit()
        done = min(i + args.chunk, len(rows))
        print(dim(f"  {done:,}/{len(rows):,} markets, {n_bars:,} bars"), flush=True)

    if args.errors_out:
        with open(args.errors_out, "w", encoding="utf-8") as fh:
            json.dump(errors, fh, indent=2)
    print(ok(f"{n_bars:,} daily bars stored; {n_empty:,} markets had no CLOB history"))
    con.close()


if __name__ == "__main__":
    main()
