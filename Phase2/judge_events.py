"""Phase 2b: classify each kept event against the Caldara-Iacoviello taxonomy.

Phase 2 answered "could this matter?" and answered it permissively: the rubric
tells the model to keep an event at a one-in-five chance of relevance, because
a false negative is permanent and a false positive is only dilution. The
measured consequence is that roughly 19 of every 20 kept events are not
geopolitical risk under the definition Caldara and Iacoviello use. That is by
design, and it is why the kept set cannot be counted, weighted, or correlated
as it stands -- a first correlation of open-event counts against GPRD over the
recent window came out at -0.39, which is a statement about listing growth and
not about the world.

This pass is the precision half. It reads the full event payload -- title,
description, tags -- and emits a structured record or a rejection.

Four fields, and each answers a question the index will otherwise have to
guess at:

  category    one of CI's eight. Not a convenience: CI publish per-category
              shares, so emitting their taxonomy buys a category-by-category
              comparison instead of a single headline number. Categories 1-5
              are threats (their GPRT), 6-8 are realized acts (their GPRA),
              and they publish those two halves daily as well.

  direction   whether a YES resolution means risk up or down. The primary
              index does not use it. CI's index is an article count, so it is
              non-negative by construction and has no way to represent priced
              de-escalation; an index that subtracts cannot be validated
              against it. The field is collected anyway because the signed
              variant is then a downstream recomputation rather than a second
              labelling run, and because "how often does de-escalation even
              appear?" is a question worth answering with a number.

  region      ISO3. CI publish per-country series, and country-level
              agreement is a far stronger result than headline agreement.

  confidence  three discrete levels, deliberately not a 0-1 score. A model
              asked for a probability returns 0.7, 0.8 and 0.9 for nearly
              everything, which is a three-level scale wearing a costume and
              it invites averaging numbers that were never measured. Three
              levels give a threshold to report sensitivity against.

One event per request, unlike Phase 2's chunks of 50. Chunking was right there
because the output was a sparse id list, where one item can barely influence
another. Here the model picks a category and a direction, and those are
exactly the judgments that anchor: eight Ukraine events in one prompt and the
ninth about Taiwan gets read through a Ukraine frame. Isolation costs about
6M extra input tokens, which at the measured batch rate is under forty cents.

Usage:
    python Phase2/judge_events.py --submit
    python Phase2/judge_events.py --collect
    python Phase2/judge_events.py --report
    python Phase2/judge_events.py --limit 40 --direct    # live sanity check
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sqlite3
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm  # noqa: E402
from Phase2.iso3 import ISO3  # noqa: E402
from common import bold, dim, err, header, info, ok, warn  # noqa: E402
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DB = os.path.join(REPO, "data", "Phase1", "Polymarket", "polymarket_raw.db")
LABEL_DB = os.path.join(REPO, "data", "Phase2", "event_filter.db")
OUT_DB = os.path.join(REPO, "data", "Phase2", "event_judge.db")
PROMPT = os.path.join(REPO, "Phase2", "prompts", "event_judge.md")

# Descriptions run to 4,400 characters, and the tail is resolution boilerplate
# -- oracle sources, UMA dispute windows, timezone rules. The subject of the
# market is always stated first. Truncating pays about 20% of input tokens for
# no measured loss of signal.
DESC_CHARS = 900

ENQUEUED_TOKEN_BUDGET = 4_000_000     # the org cap is ~5M; leave headroom
TOK_PER_CHAR = 0.25                   # rough, only used to stay under the cap
MAX_TOKENS = 200

CATEGORIES = {
    1: "war threats", 2: "peace threats", 3: "military buildups",
    4: "nuclear threats", 5: "terror threats", 6: "beginning of war",
    7: "escalation of war", 8: "terror acts",
}
DIRECTIONS = ("up", "down", "unclear")
CONFIDENCES = ("high", "medium", "low")


def prompt_text() -> str:
    with open(PROMPT) as fh:
        return fh.read().strip()


def prompt_sha() -> str:
    return hashlib.sha256(prompt_text().encode()).hexdigest()[:12]


def _flat(text) -> str:
    return " ".join(str(text or "").split())


def load_events(limit: int | None = None) -> list[dict]:
    """Every event Phase 2 kept, with the text this pass needs to judge it.

    Ordered by event id rather than shuffled. Phase 2 needed a shuffle because
    a chunk of 50 adjacent events is a run of one topic and the model
    pattern-matches down it. One event per request has no neighbours to match,
    so the order carries no risk and a stable order makes a partial run
    readable.
    """
    lab = sqlite3.connect(f"file:{LABEL_DB}?mode=ro", uri=True)
    kept = {str(r[0]) for r in lab.execute(
        "SELECT event_id FROM labels WHERE provider = 'openai' "
        "AND relevant = 1")}
    lab.close()
    if not kept:
        sys.exit(err("no kept events in the filter database"))

    src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    rows = []
    for eid, title, desc, tags in src.execute(
            "SELECT event_id, title, description, tags FROM events"):
        if str(eid) not in kept:
            continue
        rows.append({"id": str(eid), "title": _flat(title),
                     "desc": _flat(desc)[:DESC_CHARS], "tags": _tags(tags)})
    src.close()

    missing = len(kept) - len(rows)
    if missing:
        print(warn(f"  {missing:,} kept events absent from the raw catalog"))
    rows.sort(key=lambda r: int(r["id"]) if r["id"].isdigit() else 0)
    return rows[:limit] if limit else rows


def _tags(raw) -> str:
    try:
        vals = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        return ""
    out = []
    for v in vals:
        name = v.get("label") or v.get("slug") if isinstance(v, dict) else v
        # "All" is on nearly every event and separates nothing.
        if name and str(name) != "All":
            out.append(str(name))
    return " | ".join(out[:12])


def render(row: dict) -> str:
    parts = [f"TITLE: {row['title']}"]
    if row["tags"]:
        parts.append(f"TAGS: {row['tags']}")
    if row["desc"]:
        parts.append(f"DESCRIPTION: {row['desc']}")
    return "\n".join(parts)


def parse(text: str) -> tuple[dict | None, str]:
    """Validate one response. Returns (record, problem).

    Every field is checked against its closed vocabulary. A model that invents
    a ninth category or writes "United States" where an ISO3 code belongs must
    fail loudly here: a bad value that reaches the database becomes a silent
    hole in one category's series, and no downstream count can reveal it.
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return None, "no JSON object in response"
    try:
        obj = json.loads(raw[start:end + 1])
    except ValueError as exc:
        return None, f"bad JSON: {exc}"
    if not isinstance(obj, dict):
        return None, "response was not a JSON object"
    keys = {"relevant", "category", "direction", "region", "confidence"}
    if set(obj) != keys:
        return None, "response keys must exactly match the judge contract"
    if type(obj["relevant"]) is not bool:
        return None, "relevant must be a JSON boolean"
    if not obj["relevant"]:
        if any(obj[k] is not None for k in keys - {"relevant"}):
            return None, "non-relevant responses must use null fields"
        return {"relevant": 0, "category": None, "direction": None,
                "region": None, "confidence": None}, ""

    cat = obj["category"]
    if isinstance(cat, bool) or cat not in CATEGORIES:
        return None, f"category {cat!r} not in 1-8"
    direction = obj["direction"]
    if direction not in DIRECTIONS:
        return None, f"direction {direction!r} not in {DIRECTIONS}"
    conf = obj["confidence"]
    if conf not in CONFIDENCES:
        return None, f"confidence {conf!r} not in {CONFIDENCES}"
    region = obj["region"]
    if not isinstance(region, str) or region.upper() not in ISO3:
        return None, f"region {region!r} is not an ISO3 code"
    region = region.upper()
    return {"relevant": 1, "category": int(cat), "direction": direction,
            "region": region, "confidence": conf}, ""


def init_out() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(OUT_DB), exist_ok=True)
    con = sqlite3.connect(OUT_DB)
    had_judgments = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='judgments'"
    ).fetchone() is not None
    if had_judgments:
        # Released vintages are evidence, not migration targets.  In
        # particular, do not add columns or attempt tables to this database.
        return con
    con.executescript("""
        CREATE TABLE IF NOT EXISTS judgments (
            event_id TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            relevant INTEGER NOT NULL, category INTEGER, direction TEXT,
            region TEXT, confidence TEXT, prompt_sha TEXT NOT NULL,
            input_sha256 TEXT NOT NULL DEFAULT '',
            desc_chars INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (event_id, provider, model)
        );
        CREATE TABLE IF NOT EXISTS failures (
            event_id TEXT, provider TEXT, model TEXT, note TEXT,
            prompt_sha TEXT, at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS judge_attempts (
            event_id TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL,
            prompt_sha TEXT NOT NULL, path TEXT NOT NULL, attempt INTEGER NOT NULL,
            status TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, batch_id TEXT,
            request_id TEXT, note TEXT NOT NULL DEFAULT '',
            at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS ix_judg_cat ON judgments(category);
    """)
    con.commit()
    return con


def done_ids(con, provider: str, model: str) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT event_id FROM judgments WHERE provider = ? AND model = ?",
        (provider, model))}


def store(con, event_id, provider, model, rec, sha, input_text="") -> None:
    input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest() if input_text else ""
    con.execute(
        "INSERT OR REPLACE INTO judgments (event_id, provider, model, "
        "relevant, category, direction, region, confidence, prompt_sha, "
        "input_sha256, desc_chars) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (event_id, provider, model, rec["relevant"], rec["category"],
         rec["direction"], rec["region"], rec["confidence"], sha,
         input_sha, len(input_text)))


def fail(con, event_id, provider, model, note, sha) -> None:
    con.execute("INSERT INTO failures (event_id, provider, model, note, "
                "prompt_sha) VALUES (?,?,?,?,?)",
                (event_id, provider, model, note[:300], sha))


# --------------------------------------------------------------- direct path

def run_direct(args, rows, system, sha) -> None:
    model, base_url, key_env = llm.resolve(args.provider, args.model,
                                           args.base_url)
    con = init_out()
    todo = [r for r in rows if r["id"] not in done_ids(con, args.provider,
                                                       model)]
    print(info(f"{len(todo):,} events to judge, {args.workers} workers"))

    def work(row):
        try:
            text, usage = llm.complete(args.provider, model, base_url,
                                       key_env, system, render(row),
                                       max_tokens=MAX_TOKENS)
        except Exception as exc:
            return row, None, f"request failed: {exc}", {}
        rec, problem = parse(text)
        return row, rec, problem, usage

    n_ok = n_bad = tok_in = tok_out = 0
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        for row, rec, problem, usage in pool.map(work, todo):
            tok_in += usage.get("input", 0)
            tok_out += usage.get("output", 0)
            if rec is None:
                n_bad += 1
                fail(con, row["id"], args.provider, model, problem, sha)
            else:
                n_ok += 1
                store(con, row["id"], args.provider, model, rec, sha, render(row))
            con.execute(
                "INSERT INTO judge_attempts "
                "(event_id,provider,model,prompt_sha,path,attempt,status,"
                "input_tokens,output_tokens,note) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (row["id"], args.provider, model, sha, "direct", 1,
                 "ok" if rec is not None else "error",
                 usage.get("input", 0), usage.get("output", 0), problem or ""))
            if (n_ok + n_bad) % 50 == 0:
                con.commit()
    con.commit()
    con.close()


# ---------------------------------------------------------------- batch path

def _openai_request(url, api_key, data=None, method=None, ctype=None):
    hdr = {"Authorization": f"Bearer {api_key}"}
    if ctype:
        hdr["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=hdr, method=method)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def batch_stem(provider: str, model: str) -> str:
    return os.path.join(os.path.dirname(OUT_DB), f"judge_{provider}_{model}")


def _body(provider, model, system, user) -> dict:
    spec = llm.PROVIDER_DEFAULTS.get(provider, {})
    return {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            spec.get("token_field", "max_tokens"): MAX_TOKENS,
            **(spec.get("extra") or {})}


def submit_batch(args, rows, system, sha) -> None:
    if args.provider != "openai":
        raise SystemExit(
            f"batch mode is OpenAI-only; use --direct for provider '{args.provider}'"
        )
    model, base_url, key_env = llm.resolve(args.provider, args.model,
                                           args.base_url)
    api_key = os.environ.get(key_env)
    if not api_key:
        sys.exit(err(f"{key_env} is not set"))

    con = init_out()
    already = done_ids(con, args.provider, model)
    stem = batch_stem(args.provider, model)
    if os.path.exists(stem + ".batches.json"):
        sys.exit(err("a batch is already live for this provider/model.\n"
                     "  Run --collect first.\n"
                     f"  To start over, delete {stem}.batches.json"))

    todo = [r for r in rows if r["id"] not in already]
    if not todo:
        print(ok("every event already judged")); return

    # One live request before queueing thousands. Catches a bad key, an
    # unknown model id, and the max_tokens parameter split -- each of which
    # otherwise surfaces only as a wholly failed batch hours later.
    if args.dry_run:
        print(warn("DRY RUN: no API calls, no upload, no batch created"))
    else:
        print(info("preflight: one live request before submitting..."))
        try:
            text, usage = llm.complete(args.provider, model, base_url,
                                       key_env, system, render(todo[0]),
                                       max_tokens=MAX_TOKENS)
        except Exception as exc:
            sys.exit(err(f"preflight failed, nothing submitted: {exc}"))
        rec, problem = parse(text)
        if problem:
            sys.exit(err(f"preflight unusable, nothing submitted: {problem}\n"
                         f"  raw: {text[:200]}"))
        print(ok(f"  preflight ok: {rec} "
                 f"({usage.get('input', 0)} in / {usage.get('output', 0)} out)"))

    path = f"{stem}.000.jsonl"
    n, est_tok, held = 0, 0.0, 0
    with open(path, "w") as fh:
        for row in todo:
            user = render(row)
            if est_tok >= args.token_budget:
                held += 1
                continue
            fh.write(json.dumps({
                "custom_id": f"ev-{row['id']}",
                "method": "POST", "url": "/v1/chat/completions",
                "body": _body(args.provider, model, system, user),
            }) + "\n")
            est_tok += (len(system) + len(user)) * TOK_PER_CHAR + MAX_TOKENS
            n += 1

    size = os.path.getsize(path)
    print(info(f"{n:,} requests, {size / 1e6:.0f} MB, "
               f"~{est_tok / 1e6:.1f} M tokens enqueued"))
    if held:
        print(dim(f"  {held:,} events held back under the "
                  f"{args.token_budget / 1e6:.0f} M token cap; "
                  f"--collect submits the next slice once this one drains"))
    if args.dry_run:
        print(warn(f"DRY RUN: wrote {path}, nothing uploaded")); return

    with open(path, "rb") as fh:
        boundary = "----judge"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; "
                f'name="purpose"\r\n\r\nbatch\r\n--{boundary}\r\n'
                f'Content-Disposition: form-data; name="file"; '
                f'filename="{os.path.basename(path)}"\r\n'
                f"Content-Type: application/jsonl\r\n\r\n").encode()
        body += fh.read() + f"\r\n--{boundary}--\r\n".encode()
    up = _openai_request(f"{base_url}/files", api_key, body, "POST",
                         f"multipart/form-data; boundary={boundary}")
    batch = _openai_request(
        f"{base_url}/batches", api_key,
        json.dumps({"input_file_id": up["id"],
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h"}).encode(), "POST",
        "application/json")
    print(ok(f"  {os.path.basename(path)} -> batch {batch['id']} "
             f"({batch['status']})"))
    with open(stem + ".batches.json", "w") as fh:
        json.dump([batch["id"]], fh)
    print(dim("  poll with:  --collect"))


def collect_batch(args, rows, system, sha) -> None:
    if args.provider != "openai":
        raise SystemExit(
            f"batch mode is OpenAI-only; use --direct for provider '{args.provider}'"
        )
    model, base_url, key_env = llm.resolve(args.provider, args.model,
                                           args.base_url)
    api_key = os.environ.get(key_env)
    stem = batch_stem(args.provider, model)
    manifest = stem + ".batches.json"
    if not os.path.exists(manifest):
        sys.exit(err(f"no batch manifest at {manifest}; run --submit first"))
    with open(manifest) as fh:
        batch_ids = json.load(fh)

    con = init_out()
    running = []
    n_ok = n_bad = 0
    for bid in batch_ids:
        batch = _openai_request(f"{base_url}/batches/{bid}", api_key)
        counts = batch.get("request_counts", {})
        if batch["status"] not in ("completed", "failed", "expired",
                                   "cancelled"):
            print(info(f"batch {bid}: {batch['status']}  "
                       f"{counts.get('completed', 0):,}/"
                       f"{counts.get('total', 0):,} done, "
                       f"{counts.get('failed', 0)} failed"))
            running.append(bid)
            continue
        if not batch.get("output_file_id"):
            print(err(f"batch {bid}: {batch['status']}, no output file"))
            continue
        raw = urllib.request.urlopen(urllib.request.Request(
            f"{base_url}/files/{batch['output_file_id']}/content",
            headers={"Authorization": f"Bearer {api_key}"}), timeout=600)
        for line in raw.read().decode().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            eid = item["custom_id"].removeprefix("ev-")
            resp = (item.get("response") or {}).get("body") or {}
            try:
                text = resp["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                n_bad += 1
                fail(con, eid, args.provider, model, "no content", sha)
                continue
            rec, problem = parse(text)
            if rec is None:
                n_bad += 1
                fail(con, eid, args.provider, model, problem, sha)
            else:
                n_ok += 1
                store(con, eid, args.provider, model, rec, sha)
        con.commit()
    if running:
        print(dim(f"  {len(running)} batch(es) still running"))
        con.close()
        return

    print(ok(f"ingested {n_ok:,} judgments, {n_bad:,} failed"))
    os.remove(manifest)
    remaining = [r for r in rows if r["id"] not in done_ids(con,
                                                            args.provider,
                                                            model)]
    con.close()
    if remaining:
        print(info(f"{len(remaining):,} events remain; submitting next slice"))
        submit_batch(args, rows, system, sha)
    else:
        print(ok("JUDGING COMPLETE"))


def report() -> None:
    if not os.path.exists(OUT_DB):
        sys.exit(err(f"no database at {OUT_DB}"))
    con = sqlite3.connect(f"file:{OUT_DB}?mode=ro", uri=True)
    for provider, model, n, rel in con.execute(
            "SELECT provider, model, COUNT(*), SUM(relevant) FROM judgments "
            "GROUP BY provider, model"):
        print(header(f"{provider}/{model}"))
        print(f"  judged {n:,}, geopolitical {rel:,} ({100 * rel / n:.1f}%)")
    print(header("categories"))
    for cat, n in con.execute(
            "SELECT category, COUNT(*) FROM judgments WHERE relevant = 1 "
            "GROUP BY category ORDER BY category"):
        kind = "threat" if cat <= 5 else "act"
        print(f"  {cat} {CATEGORIES[cat]:<20} {kind:<7} {n:>6,}")
    for field in ("direction", "confidence"):
        print(header(field))
        for val, n in con.execute(
                f"SELECT {field}, COUNT(*) FROM judgments WHERE relevant = 1 "
                f"GROUP BY {field} ORDER BY COUNT(*) DESC"):
            print(f"  {str(val):<10} {n:>6,}")
    print(header("top regions"))
    for reg, n in con.execute(
            "SELECT region, COUNT(*) FROM judgments WHERE relevant = 1 "
            "GROUP BY region ORDER BY COUNT(*) DESC LIMIT 15"):
        print(f"  {reg} {n:>6,}")
    nf = con.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    if nf:
        print(warn(f"\n{nf:,} failures recorded"))
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--token-budget", type=float,
                    default=ENQUEUED_TOKEN_BUDGET)
    ap.add_argument("--direct", action="store_true",
                    help="live requests instead of the Batch API")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the JSONL, make no API call")
    args = ap.parse_args()
    if (args.submit or args.collect) and args.provider != "openai":
        raise SystemExit(
            f"batch mode is OpenAI-only; use --direct for provider '{args.provider}'"
        )

    if args.report:
        report(); return

    llm.load_dotenv()
    system, sha = prompt_text(), prompt_sha()
    rows = load_events(args.limit)
    print(info(f"{len(rows):,} kept events (prompt {sha})"))

    if args.submit or args.dry_run:
        submit_batch(args, rows, system, sha)
    elif args.collect:
        collect_batch(args, rows, system, sha)
    elif args.direct:
        run_direct(args, rows, system, sha)
    else:
        sys.exit(err("choose one of --submit, --collect, --direct, --report"))


if __name__ == "__main__":
    main()
