#!/usr/bin/env python3
"""Phase 1 · Collection — Polymarket contract enrichment.

Reads a SQLite database of Polymarket contracts (a filtered catalog), and for
every market fetches:
    * Gamma refresh   — volume / liquidity / outcomes / event title
    * CLOB order book — best bid/ask, spread, depth, tick size
    * hourly price history (7-day window walks over the market's lifetime)
    * client-computed daily OHLC bars from the hourly points

Writes an enriched copy to a sibling `enriched_<basename>.db` and NEVER
modifies the original.  The output database mirrors the input catalog
row-for-row and is itself the checkpoint: an existing output DB resumes by
skipping tickers whose `market_data` is already populated, so interrupted
runs continue without refetching.  For a genuinely fresh run, delete the
output file first (never done automatically).

Usage:
    python -m Phase1.enrich_contracts \
        --db data/polymarket_catalog_geo.db [-y]
"""

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (GAMMA_BASE, CLOB_BASE, HttpError,
                    accent, dim, header, highlight, info, ok, warn,
                    connect_readonly, init_db, build_url, fetch_json, post_json)

BATCH_SIZE     = 100   # markets per processing chunk
BOOK_CHUNK     = 100   # token ids per /books call (API cap, verified)
HISTORY_CHUNK  = 20    # markets per batch-prices-history call (API max)
GAMMA_CHUNK    = 20    # condition_ids per /markets refresh (URL-length guard)
WINDOW_DAYS    = 14    # verified: 14 d accepted, 21 d rejected ("interval too long")
DAY_SECONDS    = 86400


def parse_date(value: str) -> int:
    """Accept ISO date (YYYY-MM-DD) or Unix timestamp, return Unix seconds."""
    try:
        return int(value)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return int(datetime.strptime(value, fmt)
                       .replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Unrecognized date format: {value}")


def resolve_output_path(input_db: str) -> str:
    """Output path: enriched_[dbname] in the same folder."""
    db_dir = os.path.dirname(os.path.abspath(input_db))
    base = os.path.splitext(os.path.basename(input_db))[0]
    return os.path.join(db_dir, f"enriched_{base}.db")


def _parse_ts_iso(value: str | None) -> int | None:
    """Parse an ISO timestamp (with optional Z) to Unix seconds.  None/empty
    or unparseable input returns None."""
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError):
        return None


def _parse_json_list(value: str | None) -> list:
    """Parse a JSON-string field (e.g. outcomes / clobTokenIds) to a list."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def _yes_token(record: dict) -> str | None:
    toks = _parse_json_list(record.get("clobTokenIds"))
    return toks[0] if toks else None


def gamma_refresh(chunk: list[str], delay: float) -> dict[str, dict]:
    """Batch-refresh ≤100 markets via /markets?condition_ids=...&closed=<b>.
    The `closed` filter applies even with condition_ids, so call both values
    and union by conditionId.  Markets absent from both are skipped (their
    market_data stays '' and they are retried next run).
    Returns {conditionId: record}.

    condition_ids is a repeated query key, so the URL length scales with the
    batch size; 100 ids ≈ 13KB and Cloudflare's WAF began challenging that
    pathological shape from this IP.  Sub-batch to keep URLs browser-like.
    """
    refreshed: dict[str, dict] = {}
    for closed in ("true", "false"):
        for i in range(0, len(chunk), GAMMA_CHUNK):
            sub = chunk[i:i + GAMMA_CHUNK]
            params = {"condition_ids": sub, "closed": closed,
                      "limit": len(sub)}
            try:
                data = fetch_json(build_url("/markets", params,
                                            base=GAMMA_BASE))
            except HttpError as exc:
                raise RuntimeError(
                    f"Gamma /markets closed={closed} HTTP {exc.status}: "
                    f"{exc.body.decode('utf-8', 'replace')[:200]}"
                ) from exc
            for rec in data:
                cid = rec.get("conditionId")
                if cid:
                    refreshed[cid] = rec
            time.sleep(delay)
    return refreshed


def fetch_books(token_ids: list[str], delay: float) -> dict[str, dict]:
    """POST /books with up to BOOK_CHUNK token ids.  Returns
    {asset_id: book_record}."""
    books: dict[str, dict] = {}
    for i in range(0, len(token_ids), BOOK_CHUNK):
        batch = token_ids[i:i + BOOK_CHUNK]
        payload = [{"token_id": t} for t in batch]
        try:
            data = post_json(f"{CLOB_BASE}/books", payload)
        except HttpError as exc:
            raise RuntimeError(
                f"CLOB /books HTTP {exc.status}: "
                f"{exc.body.decode('utf-8', 'replace')[:200]}"
            ) from exc
        for book in data:
            aid = book.get("asset_id")
            if aid:
                books[aid] = book
        time.sleep(delay)
    return books


def book_summary(book: dict | None) -> dict:
    """Extract the snake_case book fields we store.  Missing book → all None."""
    if not book:
        return {
            "best_bid": None, "best_ask": None, "spread": None,
            "bid_depth": None, "ask_depth": None,
            "last_trade_price": None, "tick_size": None, "min_order_size": None,
        }
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    best_bid = bids[0].get("price") if bids else None
    best_ask = asks[0].get("price") if asks else None
    spread = None
    if best_bid is not None and best_ask is not None:
        try:
            spread = round(float(best_ask) - float(best_bid), 12)
        except (ValueError, TypeError):
            spread = None
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "bid_depth": sum(float(b.get("size", 0)) for b in bids),
        "ask_depth": sum(float(b.get("size", 0)) for b in asks),
        "last_trade_price": book.get("last_trade_price"),
        "tick_size": book.get("tick_size"),
        "min_order_size": book.get("min_order_size"),
    }


def fetch_history(asset_ids: list[str], start_ts: int, end_ts: int,
                  delay: float, fidelity: int = 60) -> dict[str, list[dict]]:
    """POST /batch-prices-history for ≤HISTORY_CHUNK assets over one window.
    Markets with no data in the window are omitted from the map (absent =
    no data, not an error).  Returns {asset_id: [{t, p}]}.
    Defensive: if the API rejects the window as too long (should not happen
    at ≤14 days), retry with the window halved, twice, then abort."""
    for halving in range(3):
        payload = {"markets": asset_ids, "start_ts": start_ts, "end_ts": end_ts,
                   "interval": "1m", "fidelity": fidelity}
        try:
            data = post_json(f"{CLOB_BASE}/batch-prices-history", payload)
        except HttpError as exc:
            body = exc.body.decode("utf-8", "replace")
            if exc.status == 400 and "interval is too long" in body and halving < 2:
                end_ts = start_ts + (end_ts - start_ts) // 2
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"CLOB /batch-prices-history HTTP {exc.status}: {body[:200]}"
            ) from exc
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(
                f"CLOB /batch-prices-history error: {str(data['error'])[:200]}"
            )
        return data.get("history", {}) if isinstance(data, dict) else {}
    raise RuntimeError(
        f"CLOB /batch-prices-history window still too long after halving "
        f"(start_ts={start_ts}, end_ts={end_ts})"
    )


def history_for_chunk(chunk_records: list[tuple[dict, str]],
                      history_start: int | None, now_ts: int,
                      delay: float, fidelity: int = 60) -> dict[str, list[dict]]:
    """Walk 7-day windows per market and collect hourly points.

    chunk_records: [(refreshed_record, yes_token_id_or_None), ...]
    start per market = --history-start if given, else startDate, else
    endDate − 90 days; markets with neither date skip history entirely
    (their price_history/ohlc stay '').
    Windows are walked per distinct start_ts so batch requests share a
    uniform grid (the common case is a shared --history-start).
    Returns {conditionId: [{t, p}]} sorted by t."""
    now = now_ts
    groups: dict[int, list[dict]] = {}
    for rec, tok in chunk_records:
        if not tok:
            continue
        start = history_start
        if start is None:
            start = _parse_ts_iso(rec.get("startDate"))
        if start is None:
            end_ts_raw = _parse_ts_iso(rec.get("endDate"))
            start = (end_ts_raw - 90 * DAY_SECONDS) if end_ts_raw is not None else None
        if start is None:
            continue  # no usable dates → skip history for this market
        end = _parse_ts_iso(rec.get("endDate"))
        end_limit = min(now, end + DAY_SECONDS) if end is not None else now
        if end_limit <= start:
            continue
        max_windows = math.ceil((end_limit - start) / (WINDOW_DAYS * DAY_SECONDS))
        groups.setdefault(int(start), []).append(
            {"cid": rec.get("conditionId"), "tok": tok,
             "end_limit": end_limit, "max_windows": max_windows}
        )

    points: dict[str, list[dict]] = {}
    for start_ts, members in groups.items():
        max_w = max(m["max_windows"] for m in members)
        for w in range(max_w):
            active = [m for m in members if w < m["max_windows"]]
            if not active:
                continue
            # Each request covers ONE 7-day slice.  (The API rejects windows
            # longer than ~14 days, so a cumulative end_ts would 400 on
            # w>=2.)  Clamp only the final slice to the group's end_limit.
            win_start = start_ts + w * WINDOW_DAYS * DAY_SECONDS
            win_end = start_ts + (w + 1) * WINDOW_DAYS * DAY_SECONDS
            end_ts = min(win_end, max(m["end_limit"] for m in active))
            for i in range(0, len(active), HISTORY_CHUNK):
                sub = active[i:i + HISTORY_CHUNK]
                asset_ids = [m["tok"] for m in sub]
                history = fetch_history(asset_ids, win_start, end_ts, delay, fidelity)
                for m in sub:
                    for pt in history.get(m["tok"], []):
                        points.setdefault(m["cid"], []).append(pt)
                time.sleep(delay)

    for lst in points.values():
        lst.sort(key=lambda pt: pt["t"])
    return points


def build_ohlc(points: list[dict]) -> list[dict]:
    """Daily OHLC bars from hourly points, grouped by UTC day, in order."""
    by_day: dict[str, list[float]] = {}
    for pt in points:
        day = datetime.fromtimestamp(pt["t"], tz=timezone.utc).strftime("%Y-%m-%d")
        by_day.setdefault(day, []).append(float(pt["p"]))
    bars = []
    for day in sorted(by_day):
        ps = by_day[day]
        bars.append({"date": day, "open": ps[0], "close": ps[-1],
                     "high": max(ps), "low": min(ps)})
    return bars


def build_market_data(rec: dict, book: dict) -> dict:
    """Assemble the market_data JSON object (snake_case keys)."""
    return {
        "volume":            rec.get("volume"),
        "volume24hr":        rec.get("volume24hr"),
        "volumeNum":         rec.get("volumeNum"),
        "liquidity":         rec.get("liquidity"),
        "liquidityNum":      rec.get("liquidityNum"),
        "liquidityClob":     rec.get("liquidityClob"),
        "outcomes":          _parse_json_list(rec.get("outcomes")),
        "outcomePrices":     _parse_json_list(rec.get("outcomePrices")),
        "closed":            rec.get("closed"),
        "archived":          rec.get("archived"),
        "acceptingOrders":   rec.get("acceptingOrders"),
        "umaResolutionStatuses": rec.get("umaResolutionStatuses"),
        "resolvedBy":        rec.get("resolvedBy"),
        "negRisk":           rec.get("negRisk"),
        "startDate":         rec.get("startDate"),
        "clobTokenIds":      _parse_json_list(rec.get("clobTokenIds")),
        **book_summary(book),
    }


def enrich_chunk(out_conn, chunk: list[tuple[str, str]], delay: float,
                 history_start: int | None, now_ts: int,
                 skip_history: bool = False) -> tuple[int, int]:
    """Process one chunk of (ticker, existing_sub_title) rows.
    Returns (enriched_count, history_count) for this chunk."""
    cids = [t for t, _ in chunk]
    refreshed = gamma_refresh(cids, delay)

    # Records present in the refresh, with their yes-token id.
    present: list[tuple[dict, str | None]] = []
    for t, _ in chunk:
        rec = refreshed.get(t)
        if rec is not None:
            present.append((rec, _yes_token(rec)))

    # Order books for all yes tokens in the chunk.
    token_ids = [tok for _, tok in present if tok]
    books = fetch_books(token_ids, delay)

    # Hourly history walk.  Skipped wholesale under --no-history: the CLOB
    # window walk is the bulk of both runtime and database size, and OHLC is
    # derived from its points, so both columns stay '' together.
    points = ({} if skip_history
              else history_for_chunk(present, history_start, now_ts, delay))

    # Assemble + write.
    enriched = 0
    history_written = 0
    for t, existing_sub in chunk:
        rec = refreshed.get(t)
        if rec is None:
            continue  # absent from both closed passes → retried next run
        sub_title = existing_sub
        events = rec.get("events") or []
        if events and events[0].get("title"):
            sub_title = events[0]["title"]
        md = build_market_data(rec, books.get(_yes_token(rec) or "", {}))
        description = rec.get("description") or ""
        slug = rec.get("slug") or ""
        if skip_history:
            out_conn.execute(
                "UPDATE contracts SET sub_title=?, market_data=?, description=?, "
                "slug=? WHERE ticker=?",
                (sub_title, json.dumps(md), description, slug, t),
            )
            enriched += 1
            continue
        ph = points.get(t, [])
        ohlc = build_ohlc(ph)
        out_conn.execute(
            "UPDATE contracts SET sub_title=?, market_data=?, price_history=?, ohlc=?, "
            "description=?, slug=? WHERE ticker=?",
            (sub_title, json.dumps(md), json.dumps(ph), json.dumps(ohlc),
             description, slug, t),
        )
        enriched += 1
        if ph:
            history_written += 1
    return enriched, history_written


def run(args: argparse.Namespace):
    p = lambda s: print(s, flush=True)

    if not os.path.isfile(args.db):
        p(warn(f"Error: database not found: {args.db}"))
        sys.exit(1)

    output_db = resolve_output_path(args.db)

    p("")
    p(header("=== MarketGPR  ·  Polymarket Contract Enricher ==="))
    p(accent(f"Input:  {args.db}"))
    p(accent(f"Output: {output_db}"))
    p(accent(f"Delay:  {args.delay}s/request"))
    if args.history_start:
        p(accent(f"History start: "
                 f"{datetime.fromtimestamp(args.history_start, tz=timezone.utc).strftime('%Y-%m-%d')}"))
    p("")

    if not args.yes:
        try:
            response = input(f"{highlight('Proceed?')} [y/N] ").strip().lower()
            if response not in ("y", "yes"):
                p(warn("Aborted."))
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            p("")
            p(warn("Aborted."))
            sys.exit(0)

    started = time.time()

    src = connect_readonly(args.db)
    try:
        rows = src.execute(
            "SELECT ticker, name, sub_title, event_ticker, expiry_date, fetched_at, "
            "market_data, price_history, ohlc, result, settlement_ts, volume, "
            "open_time, status, title, category, tags, last_price, description, slug "
            "FROM contracts WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
            (args.rowid_start, args.rowid_end),
        ).fetchall()
        full_cols = True
    except sqlite3.OperationalError:
        try:
            rows = src.execute(
                "SELECT ticker, name, sub_title, event_ticker, expiry_date, fetched_at "
                "FROM contracts WHERE rowid BETWEEN ? AND ? ORDER BY rowid",
                (args.rowid_start, args.rowid_end),
            ).fetchall()
            full_cols = False
        except sqlite3.OperationalError:
            src.close()
            raise RuntimeError("input database lacks a contracts table") from None
    total_rows = len(rows)

    out_conn = init_db(output_db)
    # Mirror the input catalog into the output DB (idempotent by PK); the
    # output DB is the checkpoint, so it must carry every row regardless of
    # enrichment state.  New tickers from a re-filtered input are added.
    if full_cols:
        out_conn.executemany(
            "INSERT OR IGNORE INTO contracts(ticker,name,sub_title,event_ticker,"
            "expiry_date,fetched_at,market_data,price_history,ohlc,result,"
            "settlement_ts,volume,open_time,status,title,category,tags,last_price,"
            "description,slug) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
    else:
        out_conn.executemany(
            "INSERT OR IGNORE INTO contracts(ticker,name,sub_title,event_ticker,"
            "expiry_date,fetched_at) VALUES (?,?,?,?,?,?)",
            rows,
        )
    out_conn.commit()
    if out_conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0] > 0:
        done = {
            r[0] for r in out_conn.execute(
                "SELECT ticker FROM contracts WHERE market_data != ''"
            ).fetchall()
        }
        p(info(f"Resuming: {len(done):,} already-enriched tickers in "
               f"{os.path.basename(output_db)}"))
    else:
        done = set()

    pending = [(r[0], r[2]) for r in rows if r[0] not in done]
    if args.start_offset:
        pending = pending[args.start_offset:]
    p(accent(f"Found {total_rows:,} rows  |  {len(pending):,} to enrich "
             f"(offset {args.start_offset:,})"))

    now_ts = int(time.time())
    enriched = 0
    history_written = 0
    total_chunks = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    for ci, i in enumerate(range(0, len(pending), BATCH_SIZE), start=1):
        chunk = pending[i:i + BATCH_SIZE]
        # A Cloudflare edge block (403) burns ~36 min of in-request retries
        # then raises RuntimeError — and CF blocks are per-IP, so a block
        # stalls every worker.  Retry the chunk a few times, then skip it
        # (markets stay unenriched and are picked up by a later pass)
        # instead of letting the worker die and restart into the same block.
        e = h = 0
        for attempt in range(1, 4):
            try:
                e, h = enrich_chunk(out_conn, chunk, args.delay,
                                    args.history_start, now_ts,
                                    args.no_history)
                break
            except RuntimeError as exc:
                p(warn(f"  Chunk {ci}/{total_chunks} failed (attempt "
                       f"{attempt}/3): {str(exc)[:120]}"))
                out_conn.rollback()
                if attempt < 3:
                    time.sleep(300)
        else:
            p(warn(f"  Chunk {ci}/{total_chunks} SKIPPED after 3 attempts — "
                   f"{len(chunk)} markets left for a later pass"))
            continue
        enriched += e
        history_written += h
        out_conn.commit()
        p(dim(f"  Chunk {ci}/{total_chunks}: {len(chunk)} markets  |  "
              f"{e} enriched  |  {h} with history  |  {enriched:,} total"))
    out_conn.close()
    src.close()

    out_conn = init_db(output_db)
    done_count = out_conn.execute(
        "SELECT COUNT(*) FROM contracts WHERE market_data != ''"
    ).fetchone()[0]
    total = out_conn.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
    out_conn.close()

    duration = time.time() - started
    p("")
    p(ok(f"Done: {total:,} total rows  |  {done_count:,} enriched "
         f"({history_written:,} with price history)  |  {duration:.1f}s"))
    p(ok(f"Wrote: {output_db}"))
    p("")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="enrich_contracts",
        description="Phase 1 · Collection — enrich Polymarket contracts "
                    "(Gamma metadata + description + CLOB books + hourly "
                    "history + OHLC).",
    )
    parser.add_argument("--db", required=True,
                        help="Path to input SQLite database (filtered catalog)")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Pause between API requests (seconds)")
    parser.add_argument("--history-start", type=parse_date, default=None,
                        help="Override per-market history start (ISO date or Unix timestamp)")
    parser.add_argument("--no-history", action="store_true",
                        help="Skip the CLOB price-history walk; leave "
                             "price_history and ohlc empty (much faster)")
    parser.add_argument("--start-offset", type=int, default=0,
                        help="Skip this many pending rows (rowid order) — "
                             "run several workers with disjoint offsets in "
                             "parallel against the shared output DB")
    parser.add_argument("--rowid-start", type=int, default=1,
                        help="Only load/enrich rows with rowid >= N (parallel "
                             "workers: disjoint ranges, ~150MB memory each)")
    parser.add_argument("--rowid-end", type=int, default=2**62,
                        help="Only load/enrich rows with rowid <= N")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
