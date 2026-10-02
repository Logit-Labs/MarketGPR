# MarketGPR

Code for collecting and labeling Polymarket events, fetching market prices, and constructing the MarketGPR index. This repository contains no datasets, generated outputs, or credentials.

## Setup

```sh
python -m pip install -r requirements.txt
cp .env.example .env
```

Set a provider key in `.env` only if running Phase 2. The template supports Anthropic, DeepSeek, OpenAI, and Z.ai. `.env` and `data/` are ignored by Git.

## Pipeline

1. Phase 1 collects the raw market and event catalogs. The full ID walk is large; inspect the plan first and use `--limit` for a small pull.

   ```sh
   python -m Phase1.collect_raw --dry-run
   python -m Phase1.collect_raw
   python -m Phase1.collect_events_raw
   python -m Phase1.validate_raw
   ```

   The default database is `data/Phase1/Polymarket/polymarket_raw.db`. `Phase1/enrich_contracts.py` is an optional enrichment pass for a supplied catalog.

2. Phase 2 filters and judges events with an LLM. Direct calls can incur provider charges; batch submit/collect is available with OpenAI.

   ```sh
   python Phase2/filter_events.py --provider openai --limit 10 --direct
   python Phase2/judge_events.py --provider openai --limit 10 --direct
   ```

   These limited commands are pilots, not a full corpus run. Outputs default to `data/Phase2/event_filter.db` and `data/Phase2/event_judge.db`.

3. Phase 3 fetches daily bars for judged markets. Inspect pending work before a live request.

   ```sh
   python -m Phase3.collect_judged --dry-run --limit 5
   python -m Phase3.collect_judged --limit 5
   ```

   The default output is `data/Phase3/Polymarket/judged_ohlc.db`.

4. Fetch the CI GPR comparator and run index construction:

   ```sh
   python analysis/fetch_ci_gpr.py
   python analysis/index_pm.py
   ```

## Optional data downloads

- `python analysis/ucdp.py --fetch` downloads UCDP Candidate Events data for validation. No UCDP data is included; its datasets are distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and require attribution. See the [UCDP download terms](https://ucdp.uu.se/downloads).
- `python analysis/benchmark/data.py` builds the optional external-comparator panel. It downloads and caches source series under ignored `data/`; CI monthly data is also required.
