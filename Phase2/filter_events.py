"""Phase 2: filter the Polymarket event corpus into useful / not useful.

The whole of Phase 2 is this one binary decision. It discards the ~97% of
events that are crypto candles, sports fixtures and weather, and keeps what a
geopolitical risk index could plausibly be built from. Nothing here scores,
tiers, or ranks what survives -- analysis lives downstream, and a filter that
quietly starts grading its output is a filter whose recall can no longer be
measured against a single number.

Three design choices carry most of the weight, and each is a response to a
measured property of this corpus rather than a general preference:

**Chunking (default 50 events per request).** Each item is ~31 tokens of
title and tags against an ~800-token rubric, so a request per item would
spend 96% of its input budget re-sending the rubric. Chunking cuts input
tokens by ~71%.

**Shuffling.** Events arrive in id order, which is also topic order: the
corpus contains runs of tens of thousands of near-identical crypto candles.
A model that has just emitted forty consecutive NOT_RELEVANT labels will
pattern-match the forty-first, and a geopolitical event sitting inside such a
run gets swept along with it. Shuffling breaks the run. This is not a
nice-to-have; it is the main quality control here.

**Sparse output.** The model returns only the ids it marked relevant, plus a
count. At a ~1-3% positive rate this cuts output tokens ~10x, and output is
priced 5-6x input on every provider we use, so it is the single largest cost
lever. The cost is that we cannot prove per-item that all fifty were
considered; `n_seen` and an id-subset check are weak substitutes.

On aggregating several labellers: the two errors are not symmetric -- a false
negative is permanent and undetectable, a false positive merely dilutes the
kept set -- which argues for OR-union over majority vote. Measured, that
argument did not survive contact with a weak second labeller: glm-4.5-flash
agreed with gpt-5.6-luna at kappa 0.646, and its exclusive hits were
dominated by items this prompt excludes by name. Union only pays when the
labellers are of comparable strength. See `check_agreement.py`.

These LLM labels are no longer the intended production filter -- that is a
deterministic classifier, for cost and speed -- but they are its training and
evaluation set, and the control it gets measured against.

Usage:

    # measure the positive rate before committing to the full corpus (~$0.08)
    python Phase2/filter_events.py --provider openai --limit 20000 --direct

    # full corpus via the OpenAI Batch API (-50%), submit then poll
    python Phase2/filter_events.py --provider openai --submit
    python Phase2/filter_events.py --provider openai --collect

    # a second labeller, free tier, no batch endpoint
    python Phase2/filter_events.py --provider zai --direct --workers 2

    # what survived, across every provider that has run
    python Phase2/filter_events.py --report
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402
from common import bold, dim, err, header, info, ok, warn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DB = os.path.join(REPO, "data", "Phase1", "Polymarket", "polymarket_raw.db")
OUT_DB = os.path.join(REPO, "data", "Phase2", "event_filter.db")
PROMPT = os.path.join(REPO, "Phase2", "prompts", "event_filter.md")

CHUNK = 50
SHUFFLE_SEED = 20260906

# Two independent caps bind here, and the token one binds much harder:
#   * an input file may be 200 MB / 50,000 requests
#   * the ORG may hold only ~5,000,000 *enqueued* batch tokens at once
# The corpus is ~45 M tokens, so it cannot be submitted in one go at any file
# size. Work is therefore sent one token-bounded shard at a time, and the next
# shard goes out only when the previous has drained -- --collect does that,
# so the workflow is: --submit once, then --collect periodically until done.
MAX_SHARD_BYTES = 80_000_000
MAX_SHARD_REQS = 20_000
ENQUEUED_TOKEN_BUDGET = 4_000_000     # under the 5 M cap, leaving headroom
TOK_PER_CHAR = 0.25                   # rough; only used to stay under the cap

# --------------------------------------------------------------------------
# scopes
#
# A scope is a named sub-corpus, chunked and numbered independently of the
# full run. It exists because the corpus is not uniformly worth labelling:
# 64% of events closed after March 2026, which is mostly the hourly crypto
# candle spam, while the 2025-04..2026-03 window is the only stretch where
# Polymarket persisted the volume window fields (volume1wk / 1mo / 1yr) at
# market close. Those frozen values are the project's only ground truth for
# real dollar volume, so that window earns the labelling budget first.
#
# Scope bounds are close-month strings compared as text, [lo, hi).
SCOPES = {
    # Polymarket began persisting the volume windows at close in ~2025-04 and
    # stopped emitting them after ~2026-03. Both cliffs were measured against
    # close date over all 2.87 M markets, and neither tracks market age.
    "recent": ("2025-04", "2026-04", True),
    # The two sides of the recent window. Labelling them brings the corpus to
    # uniform coverage, which matters more than it sounds: an index computed
    # across a coverage boundary reads a 4x step in our own sampling rate as
    # if it were a change in the world.
    "back": ("0000", "2025-04", True),
    # Forward events are the live tail of the corpus, so `closed = 1` would
    # drop most of them. An open market still has a price series, which is
    # all the index needs.
    "fwd": ("2026-04", "9999", False),
}

# Chunk ids are positions in a shuffled corpus, so a scope must not reuse the
# full run's numbering or `runs` rows would collide and --collect would match
# a returned batch to the wrong events. One offset per scope, far above the
# full corpus's 18,613 chunks.
CHUNK_NS = {"recent": 1_000_000, "back": 2_000_000,
            "fwd": 3_000_000}
CHUNK_OFFSET = 0                      # set by load_events(); read by chunks()


# --------------------------------------------------------------------------
# corpus


def load_events(limit: int | None, scope: str | None = None,
                skip_labelled: tuple[str, str] | None = None
                ) -> list[tuple[str, str, str]]:
    """(event_id, title, tags) for the corpus, shuffled deterministically.

    The seed is fixed so a resumed or re-run job rebuilds the identical
    chunks -- otherwise `--collect` could not match a batch's results back to
    the events that produced it, and a partial run could not be topped up.

    `scope` restricts the corpus to one named close-date window and shifts
    the chunk numbering into that scope's namespace. Determinism has to hold
    inside a scope too, and that is the awkward part: a scope wants to skip
    events some earlier pass already labelled, but the label set grows while
    the scope is running, so recomputing the skip on every invocation would
    reshuffle the chunks under a live batch. The membership is therefore
    frozen to a manifest on first build and reused verbatim after that.
    """
    global CHUNK_OFFSET
    con = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    if scope:
        lo, hi, want_closed = SCOPES[scope]
        rows = con.execute(
            "SELECT event_id, title, COALESCE(tags,''), "
            "       substr(COALESCE(closed_time, end_date, ''), 1, 7) AS m "
            "FROM events WHERE title IS NOT NULL AND title != '' "
            f"  {'AND closed = 1' if want_closed else ''} "
            "  AND m >= ? AND m < ? ORDER BY event_id", (lo, hi)).fetchall()
        rows = [r[:3] for r in rows]
    else:
        rows = con.execute(
            "SELECT event_id, title, COALESCE(tags,'') FROM events "
            "WHERE title IS NOT NULL AND title != '' ORDER BY event_id").fetchall()
    con.close()
    # Tags are stored as a JSON array. Rendering them as one costs roughly
    # double the tokens of a pipe-joined list -- brackets, quotes and
    # ", " separators, on a field billed 930,634 times -- and buys nothing.
    rows = [(str(a), b, _tags(c)) for a, b, c in rows]

    if scope:
        CHUNK_OFFSET = CHUNK_NS[scope]
        rows = _freeze_scope(scope, rows, skip_labelled)
    random.Random(SHUFFLE_SEED).shuffle(rows)
    return rows[:limit] if limit else rows


def _freeze_scope(scope: str, rows: list, skip_labelled) -> list:
    """Pin a scope's membership on first build; reuse it on every later run.

    Returns `rows` filtered to the frozen id set. The manifest is small (one
    id per member) and it is the reason a scope can be topped up across days
    while events outside it keep getting labelled.
    """
    path = os.path.join(os.path.dirname(OUT_DB), f"scope_{scope}.json")
    try:
        with open(path) as fh:
            keep = set(json.load(fh))
        print(dim(f"  scope {scope}: {len(keep):,} events "
                  f"(frozen, {os.path.basename(path)})"))
    except OSError:
        keep = {r[0] for r in rows}
        if skip_labelled:
            provider, model = skip_labelled
            con = sqlite3.connect(OUT_DB) if os.path.exists(OUT_DB) else None
            if con is not None:
                done = {r[0] for r in con.execute(
                    "SELECT event_id FROM labels WHERE provider=? AND model=?",
                    (provider, model))}
                con.close()
                before = len(keep)
                keep -= done
                print(dim(f"  scope {scope}: {before:,} in window, "
                          f"{before - len(keep):,} already labelled by "
                          f"{provider}/{model}, {len(keep):,} to do"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(sorted(keep), fh)
        print(dim(f"  scope {scope}: membership frozen to "
                  f"{os.path.basename(path)}"))
    return [r for r in rows if r[0] in keep]


def _tags(raw: str) -> str:
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, list):
        return "|".join(str(t) for t in parsed if t)
    return str(parsed)


def chunks(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield CHUNK_OFFSET + i // size, rows[i:i + size]


def render(chunk: list[tuple[str, str, str]]) -> str:
    """One line per event. Tags are already a '|'-joined string upstream.

    Deliberately terse: every character here is multiplied by 930,634.
    """
    lines = []
    for eid, title, tags in chunk:
        tag = f"  [{_flat(tags)}]" if tags else ""
        lines.append(f"{eid}\t{_flat(title)}{tag}")
    return (f"{len(chunk)} items follow, one per line, "
            f"as: id<TAB>title  [tags]\n\n" + "\n".join(lines))


def _flat(text: str) -> str:
    """Collapse tabs/newlines. None appear in the corpus today, but one would
    silently break the line-per-item contract and shift every id after it --
    a corruption the id-subset check cannot catch."""
    return " ".join(str(text).split())


def prompt_text() -> str:
    with open(PROMPT) as fh:
        return fh.read()


# --------------------------------------------------------------------------
# storage


def init_out() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)
    con = sqlite3.connect(OUT_DB)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS labels(
            event_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            model    TEXT NOT NULL,
            relevant INTEGER NOT NULL,
            PRIMARY KEY (event_id, provider, model));
        CREATE TABLE IF NOT EXISTS runs(
            chunk_id  INTEGER NOT NULL,
            provider  TEXT NOT NULL,
            model     TEXT NOT NULL,
            n_sent    INTEGER NOT NULL,
            n_seen    INTEGER,
            n_hit     INTEGER,
            status    TEXT NOT NULL,
            note      TEXT NOT NULL DEFAULT '',
            in_tok    INTEGER DEFAULT 0,
            out_tok   INTEGER DEFAULT 0,
            prompt_sha TEXT NOT NULL DEFAULT '',
            ts        TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (chunk_id, provider, model));
        CREATE INDEX IF NOT EXISTS idx_lab_rel ON labels(relevant);
    """)
    return con


def done_chunks(con, provider: str, model: str) -> set[int]:
    return {r[0] for r in con.execute(
        "SELECT chunk_id FROM runs WHERE provider=? AND model=? "
        "AND status='ok'", (provider, model))}


# --------------------------------------------------------------------------
# response handling


def parse(text: str, sent: list[str]) -> tuple[list[str], int, str]:
    """Return (relevant_ids, n_seen, problem).

    Validates that every returned id was actually in the request. A model
    that invents or mangles an id is a silent data-corruption source at this
    scale, so an unknown id fails the chunk rather than being dropped
    quietly.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if "```" in body[3:] else body[3:]
        body = body.split("\n", 1)[-1] if body[:4].isalpha() else body
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        # Fall back to the outermost {...}: cheap models prepend prose even
        # when asked for bare JSON.
        i, j = body.find("{"), body.rfind("}")
        if i < 0 or j <= i:
            return [], -1, "no JSON object in response"
        try:
            obj = json.loads(body[i:j + 1])
        except json.JSONDecodeError as exc:
            return [], -1, f"unparseable: {exc}"
    if not isinstance(obj, dict):
        return [], -1, "response was not a JSON object"
    if set(obj) != {"maybe_relevant", "n_seen"}:
        return [], -1, "response keys must be exactly maybe_relevant and n_seen"

    hits = obj["maybe_relevant"]
    if not isinstance(hits, list):
        return [], -1, "maybe_relevant missing or not a list"
    hits = [str(x) for x in hits]

    allowed = set(sent)
    unknown = [h for h in hits if h not in allowed]
    if unknown:
        return [], -1, f"{len(unknown)} id(s) not in request, e.g. {unknown[:3]}"

    n_seen = obj["n_seen"]
    if isinstance(n_seen, bool) or not isinstance(n_seen, int):
        return [], -1, "n_seen must be an integer"
    if not 0 <= n_seen <= len(sent):
        return [], -1, f"n_seen={n_seen} outside 0..{len(sent)}"
    return sorted(set(hits)), n_seen, ""


def store(con, chunk_id, provider, model, sent, hits, n_seen, usage, sha):
    hit = set(hits)
    con.executemany(
        "INSERT OR REPLACE INTO labels VALUES(?,?,?,?)",
        [(eid, provider, model, 1 if eid in hit else 0) for eid in sent])
    note = "" if n_seen == len(sent) else f"n_seen={n_seen} sent={len(sent)}"
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
        (chunk_id, provider, model, len(sent), n_seen, len(hit), "ok", note,
         usage.get("input", 0), usage.get("output", 0), sha))
    con.commit()


def fail(con, chunk_id, provider, model, n_sent, note, sha):
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
        (chunk_id, provider, model, n_sent, None, None, "failed", note[:300],
         0, 0, sha))
    con.commit()


# --------------------------------------------------------------------------
# direct (synchronous) mode


def run_direct(args, rows, system, sha):
    model, base_url, key_env = llm.resolve(args.provider, args.model,
                                           args.base_url)
    con = init_out()
    already = done_chunks(con, args.provider, model)
    todo = [(cid, ch) for cid, ch in chunks(rows, args.chunk)
            if cid not in already]
    if not todo:
        print(ok("nothing to do; every chunk already succeeded"))
        return
    print(info(f"{len(todo):,} chunks to run "
               f"({len(already):,} already done) on {args.provider}/{model}"))

    tally = {"in": 0, "out": 0, "hits": 0, "fail": 0}

    def work(item):
        cid, chunk = item
        sent = [r[0] for r in chunk]
        try:
            text, usage = llm.complete(args.provider, model, base_url, key_env,
                                       system, render(chunk),
                                       max_tokens=args.max_tokens)
        except Exception as exc:
            return cid, chunk, sent, None, None, {}, str(exc)
        hits, n_seen, problem = parse(text, sent)
        if not problem and n_seen != len(sent):
            problem = f"n_seen={n_seen} sent={len(sent)}"
        return cid, chunk, sent, hits, n_seen, usage, problem

    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(work, todo), 1):
            cid, chunk, sent, hits, n_seen, usage, problem = res
            if problem:
                tally["fail"] += 1
                fail(con, cid, args.provider, model, len(sent), problem, sha)
                print(err(f"  chunk {cid}: {problem}"))
                continue
            store(con, cid, args.provider, model, sent, hits, n_seen, usage, sha)
            tally["in"] += usage.get("input", 0)
            tally["out"] += usage.get("output", 0)
            tally["hits"] += len(hits)
            if i % 20 == 0 or i == len(todo):
                print(dim(f"  {i:,}/{len(todo):,} chunks  "
                          f"hits={tally['hits']:,}  fail={tally['fail']}  "
                          f"tok={tally['in']:,}in/{tally['out']:,}out"))
    con.close()
    print(ok(f"done: {tally['hits']:,} relevant, {tally['fail']} failed chunks"))


class _DryRun(Exception):
    """Short-circuits the preflight so --dry-run exercises every code path
    that does not cost money. The absence of this was how a `--submit` meant
    as a smoke test enqueued the entire corpus."""


# --------------------------------------------------------------------------
# OpenAI Batch API mode (-50%, results within 24h)


def _body_params(provider: str, max_tokens: int) -> dict:
    """Mirror llm.complete's per-provider request shape into the batch JSONL.

    A batch cannot negotiate: the direct path can retry a 400 with the other
    output-cap parameter, but 18,613 queued requests all fail together. So the
    shape has to be right the first time, and it has to be the same shape the
    preflight probe validated.
    """
    spec = llm.PROVIDER_DEFAULTS.get(provider, {})
    return {spec.get("token_field", "max_tokens"): max_tokens,
            **(spec.get("extra") or {})}


def _openai_request(url, api_key, data=None, method=None, ctype=None):
    hdr = {"Authorization": f"Bearer {api_key}"}
    if ctype:
        hdr["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def batch_stem(args, model: str) -> str:
    """Path stem for a run's batch manifest and JSONL shards.

    A scope gets its own stem. Without that, a scoped submit would overwrite
    the full run's .batches.json and make the earlier job's results
    unreachable -- the exact failure the guard in submit_batch prevents
    between two runs of the same shape.
    """
    name = f"batch_{args.provider}_{model}"
    if getattr(args, "scope", None):
        name += f"_{args.scope}"
    return os.path.join(os.path.dirname(OUT_DB), name)


def submit_batch(args, rows, system, sha):
    if args.provider != "openai":
        raise SystemExit(
            f"batch mode is OpenAI-only; use --direct for provider '{args.provider}'"
        )
    """Write a JSONL of chunk requests, upload it, create the batch.

    The batch id is recorded on disk so --collect can find it without the
    user having to carry it between invocations.
    """
    model, base_url, key_env = llm.resolve(args.provider, args.model,
                                           args.base_url)
    api_key = os.environ.get(key_env)
    if not api_key:
        sys.exit(err(f"{key_env} is not set"))

    con = init_out()
    already = done_chunks(con, args.provider, model)
    stem = batch_stem(args, model)

    # Refuse to pile a second submission on top of a live one: the batch ids
    # would be overwritten and the first job's results become unreachable.
    if os.path.exists(stem + ".batches.json"):
        sys.exit(err("a batch is already submitted for this provider/model.\n"
                     "  Run --collect first; it ingests results and leaves "
                     "only failed chunks to resubmit.\n"
                     f"  To start over, delete {stem}.batches.json"))

    if args.dry_run:
        print(warn("DRY RUN: no API calls, no upload, no batch created"))
    # One live request before writing ~200 MB and waiting up to 24h. Catches
    # a bad key, an unknown model id, and the max_tokens/max_completion_tokens
    # split -- all of which otherwise surface as a wholly failed batch.
    probe_id, probe = next(chunks(rows, args.chunk))
    if not args.dry_run:
        print(info("preflight: one live request before submitting..."))
    try:
        if args.dry_run:
            raise _DryRun
        text, usage = llm.complete(args.provider, model, base_url, key_env,
                                   system, render(probe),
                                   max_tokens=args.max_tokens)
    except _DryRun:
        pass
    except Exception as exc:
        sys.exit(err(f"preflight failed, nothing submitted: {exc}"))
    else:
        hits, n_seen, problem = parse(text, [r[0] for r in probe])
        if problem:
            sys.exit(err(f"preflight response unusable, nothing submitted: "
                         f"{problem}\n  raw: {text[:200]}"))
        print(ok(f"  preflight ok: {len(hits)}/{len(probe)} relevant, "
                 f"{usage.get('input',0)} in / {usage.get('output',0)} out"))

    n = 0
    est_tok = 0.0
    shard, shard_bytes, shard_n = 0, 0, 0
    paths = []
    submitted_chunks = []
    stopped_early = False
    fh = open(f"{stem}.000.jsonl", "w"); paths.append(f"{stem}.000.jsonl")
    for cid, chunk in chunks(rows, args.chunk):
            if cid in already:
                continue
            submitted_chunks.append([cid, [r[0] for r in chunk]])
            if est_tok >= args.token_budget:
                stopped_early = True
                break
            line = json.dumps({
                "custom_id": f"chunk-{cid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    **_body_params(args.provider, args.max_tokens),
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": render(chunk)}],
                },
            }) + "\n"
            if shard_bytes + len(line) > MAX_SHARD_BYTES or shard_n >= MAX_SHARD_REQS:
                fh.close()
                shard += 1
                shard_bytes, shard_n = 0, 0
                paths.append(f"{stem}.{shard:03d}.jsonl")
                fh = open(paths[-1], "w")
            fh.write(line)
            shard_bytes += len(line)
            shard_n += 1
            n += 1
            est_tok += len(line) * TOK_PER_CHAR
    fh.close()
    paths = [q for q in paths if os.path.getsize(q) > 0]
    if not n:
        for q in paths:
            os.remove(q)
        print(ok("nothing to submit")); return
    total_mb = sum(os.path.getsize(q) for q in paths) / 1e6
    remaining = sum(1 for cid, _ in chunks(rows, args.chunk)
                    if cid not in already) - n
    print(info(f"{n:,} chunk requests across {len(paths)} shard(s), "
               f"{total_mb:.0f} MB, ~{est_tok/1e6:.1f} M tokens enqueued"))
    if stopped_early:
        print(dim(f"  {remaining:,} chunks held back to stay under the "
                  f"{args.token_budget/1e6:.0f} M enqueued-token cap;\n"
                  f"  --collect submits the next slice once this one drains"))

    if args.dry_run:
        for q in paths:
            print(dim(f"  would upload {os.path.basename(q)} "
                      f"({os.path.getsize(q)/1e6:.1f} MB)"))
            os.remove(q)
        print(warn("DRY RUN complete: nothing was submitted, shards removed"))
        return

    batch_ids = []
    for q in paths:
        boundary = "----gprk" + hashlib.md5(q.encode()).hexdigest()[:16]
        with open(q, "rb") as fh:
            payload = fh.read()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\""
            f"\r\n\r\nbatch\r\n--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"file\"; filename=\"{os.path.basename(q)}\"\r\n"
            f"Content-Type: application/jsonl\r\n\r\n").encode() + payload + \
            f"\r\n--{boundary}--\r\n".encode()
        up = _openai_request("https://api.openai.com/v1/files", api_key, body,
                             "POST", f"multipart/form-data; boundary={boundary}")
        batch = _openai_request(
            "https://api.openai.com/v1/batches", api_key,
            json.dumps({"input_file_id": up["id"],
                        "endpoint": "/v1/chat/completions",
                        "completion_window": "24h"}).encode(),
            "POST", "application/json")
        batch_ids.append(batch["id"])
        print(ok(f"  {os.path.basename(q)} -> batch {batch['id']} "
                 f"({batch['status']})"))
    with open(stem + ".batches.json", "w") as fh:
        json.dump(batch_ids, fh)
    h = hashlib.sha256()
    with open(SRC_DB, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    with open(stem + ".identity.json", "w") as fh:
        json.dump({
            "prompt_sha": sha, "catalog_sha256": h.hexdigest(),
            "chunks": submitted_chunks, "provider": args.provider,
            "model": model, "base_url": base_url, "batch_ids": batch_ids,
        }, fh, indent=2)
    print(dim("  poll with:  --collect"))


def collect_batch(args, rows, system, sha):
    if args.provider != "openai":
        raise SystemExit(
            f"batch mode is OpenAI-only; use --direct for provider '{args.provider}'"
        )
    model, base_url, key_env = llm.resolve(args.provider, args.model, args.base_url)
    stem = batch_stem(args, model)
    try:
        with open(stem + ".batches.json") as fh:
            batch_ids = json.load(fh)
        with open(stem + ".identity.json") as fh:
            identity = json.load(fh)
    except OSError:
        sys.exit(err("no submitted batch identity on disk; run --submit first"))
    h = hashlib.sha256()
    with open(SRC_DB, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    current = {
        "prompt_sha": sha, "catalog_sha256": h.hexdigest(),
        "model": model, "base_url": base_url,
    }
    for key, value in current.items():
        if identity.get(key) != value:
            raise SystemExit(
                f"batch identity mismatch: {key} recorded={identity.get(key)!r} "
                f"current={value!r}"
            )
    payloads, pending = [], 0
    for bid in batch_ids:
        batch = _openai_request(f"https://api.openai.com/v1/batches/{bid}",
                                api_key)
        counts = batch.get("request_counts", {})
        print(info(f"batch {bid}: {batch['status']}  "
                   f"{counts.get('completed',0):,}/{counts.get('total',0):,} done, "
                   f"{counts.get('failed',0):,} failed"))
        if batch["status"] != "completed":
            pending += 1
            continue
        if not batch.get("output_file_id"):
            print(err(f"  {bid} completed with no output file")); continue
        req = urllib.request.Request(
            f"https://api.openai.com/v1/files/{batch['output_file_id']}/content",
            headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=900) as resp:
            payloads.append(resp.read().decode())
    if pending:
        print(dim(f"  {pending} batch(es) still running; "
                  f"re-run --collect later to pick them up"))
    if not payloads:
        return

    by_chunk = {cid: ch for cid, ch in chunks(rows, args.chunk)}
    con = init_out()
    good = bad = hits = orphan = 0
    for line in "\n".join(payloads).splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = int(rec["custom_id"].split("-")[1])
        chunk = by_chunk.get(cid)
        if chunk is None:
            # Chunk ids are positions in the shuffled corpus, so they only
            # mean the same thing when --limit matches the submitting run.
            orphan += 1
            continue
        sent = [r[0] for r in chunk]
        resp_body = (rec.get("response") or {}).get("body") or {}
        if rec.get("error") or not resp_body:
            bad += 1
            fail(con, cid, args.provider, model, len(sent),
                 json.dumps(rec.get("error"))[:300], sha)
            continue
        text = resp_body["choices"][0]["message"]["content"]
        usage = resp_body.get("usage") or {}
        usage = {"input": usage.get("prompt_tokens", 0),
                 "output": usage.get("completion_tokens", 0)}
        found, n_seen, problem = parse(text, sent)
        if not problem and n_seen != len(sent):
            problem = f"n_seen={n_seen} sent={len(sent)}"
        if problem:
            bad += 1
            fail(con, cid, args.provider, model, len(sent), problem, sha)
            continue
        store(con, cid, args.provider, model, sent, found, n_seen, usage, sha)
        good += 1
        hits += len(found)
    print(ok(f"ingested {good:,} chunks ({hits:,} relevant), {bad:,} failed"))
    # Every submitted batch is terminal, so the queue is free: send the next
    # slice automatically rather than making the user notice and re-run.
    if not pending:
        os.remove(stem + ".batches.json")
        left = sum(1 for cid, _ in chunks(rows, args.chunk)
                   if cid not in done_chunks(con, args.provider, model))
        if left and not args.no_advance:
            print(info(f"{left:,} chunks remain; submitting the next slice"))
            submit_batch(args, rows, system, sha)
        elif left:
            print(dim(f"{left:,} chunks remain; re-run --submit when ready"))
        else:
            print(ok("corpus complete"))
    if orphan:
        print(err(f"  {orphan:,} result(s) had no matching chunk -- almost "
                  f"certainly a --limit mismatch with the submitting run.\n"
                  f"  Re-run --collect with the same --limit (or none)."))
    if bad and not pending:
        print(dim("  delete the .batches.json, then --submit to retry "
                  "only the failed chunks"))


# --------------------------------------------------------------------------
# report


def report():
    con = init_out()
    print(header("Phase 2 event filter"))
    rows = con.execute(
        "SELECT provider, model, COUNT(*), SUM(relevant) FROM labels "
        "GROUP BY provider, model ORDER BY provider, model").fetchall()
    if not rows:
        print(warn("no labels yet")); return
    print(f"{'provider/model':34s}{'labelled':>12s}{'relevant':>12s}{'rate':>9s}")
    for prov, model, n, rel in rows:
        print(f"{prov+'/'+model:34s}{n:>12,}{rel or 0:>12,}"
              f"{100*(rel or 0)/n:>8.2f}%")

    u = con.execute("SELECT COUNT(DISTINCT event_id) FROM labels "
                    "WHERE relevant=1").fetchone()[0]
    tot = con.execute("SELECT COUNT(DISTINCT event_id) FROM labels").fetchone()[0]
    print(bold(f"\nUNION across providers (kept): {u:,} of {tot:,} "
               f"= {100*u/max(tot,1):.2f}%"))

    n_prov = len(rows)
    if n_prov > 1:
        both = con.execute(
            "SELECT COUNT(*) FROM (SELECT event_id FROM labels WHERE relevant=1 "
            "GROUP BY event_id HAVING COUNT(*)=?)", (n_prov,)).fetchone()[0]
        print(f"  flagged by ALL {n_prov} labellers: {both:,}")
        print(f"  contested (some but not all): {u-both:,}"
              + dim("   <- the set worth hand-labelling"))

    f = con.execute("SELECT COUNT(*) FROM runs WHERE status='failed'").fetchone()[0]
    i, o = con.execute("SELECT COALESCE(SUM(in_tok),0), COALESCE(SUM(out_tok),0) "
                       "FROM runs").fetchone()
    print(f"\ntokens: {i:,} in / {o:,} out"
          + (err(f"   failed chunks: {f:,}") if f else ""))


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default="openai",
                    choices=sorted(llm.PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N shuffled events (pilot runs)")
    ap.add_argument("--scope", default=None, choices=sorted(SCOPES),
                    help="label only one named close-date window, chunked "
                         "and numbered in its own namespace")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--token-budget", type=float, default=ENQUEUED_TOKEN_BUDGET,
                    help="stay under the org's enqueued-batch-token cap")
    ap.add_argument("--no-advance", action="store_true",
                    help="--collect: ingest only, do not submit the next slice")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--direct", action="store_true",
                      help="synchronous requests; any provider")
    mode.add_argument("--submit", action="store_true",
                      help="create an OpenAI Batch job (-50%%)")
    mode.add_argument("--collect", action="store_true",
                      help="poll and ingest a submitted batch")
    mode.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="--submit: build shards and report, but call nothing")
    args = ap.parse_args()
    if (args.submit or args.collect) and args.provider != "openai":
        raise SystemExit(
            f"batch mode is OpenAI-only; use --direct for provider '{args.provider}'"
        )

    if args.report:
        report(); return

    system = prompt_text()
    sha = hashlib.sha256(system.encode()).hexdigest()[:12]
    model_id = llm.resolve(args.provider, args.model, args.base_url)[0]
    rows = load_events(args.limit, args.scope,
                       skip_labelled=(args.provider, model_id))
    n_chunks = (len(rows) + args.chunk - 1) // args.chunk
    scope_note = f" [scope {args.scope}]" if args.scope else ""
    print(info(f"{len(rows):,} events -> {n_chunks:,} chunks of {args.chunk} "
               f"(prompt {sha}){scope_note}"))

    if args.submit:
        submit_batch(args, rows, system, sha)
    elif args.collect:
        collect_batch(args, rows, system, sha)
    else:
        run_direct(args, rows, system, sha)


if __name__ == "__main__":
    main()
