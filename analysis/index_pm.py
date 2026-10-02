"""Build the Polymarket geopolitical risk index and test it against CI.

The construction is a FLOW of belief revisions, not a level of beliefs. That
choice is the core of the design, so the reasoning is worth stating.

GPRD is not a stock. It is the count of articles published on day t: a daily
flow of attention that is non-negative and persistent because news events
persist. The belief-side analogue of a flow of attention is a flow of belief
revision -- how much probability moved today -- not a cross-sectional average
of probability levels.

A level index fails three ways here, and each failure is mechanical rather
than statistical:

  Dead markets accumulate. A "by date" market nobody trades sits forward-
  filled near 0.02 until its deadline. In a mean of levels those pile up and
  drag the index down as the universe ages. In a flow they contribute exactly
  zero, with no dead-market filter and no threshold to defend.

  Theta decay. For any "will X happen by D" market, p falls toward zero as D
  approaches even with no news at all. A level index therefore drifts down
  between news days, at a rate set by the deadline mix -- composition again.

  It answers a question CI does not ask. Mean p is "how likely is bad stuff,
  averaged over what Polymarket chose to list". GPRD is "how much is being
  written". No transform of the first yields the second.

The resolution jump is kept, deliberately. When the US struck Iran the market
did not stop moving -- it made the single largest move a market can make,
from about 0.40 to 1.00, on the day CI peaked. Dropping resolution days would
discard the act half of the signal, which is precisely the half that maps to
CI's GPRA.

That mapping is the sharpest test available. CI publish GPRT (categories 1-5,
threats) and GPRA (categories 6-8, acts) daily. Threats correspond to
revisions in still-open markets; acts correspond to resolution jumps. The
diagonal of that 2x2 dominating the off-diagonal is a far stronger result
than one aggregate correlation.

The shipped precision-tuned form uses raw probability and a capped
intraday excursion. A move is credited fully when its close-to-close change
is large, but a round trip receives at most twice that net move. This keeps
genuine revisions while removing the largest isolated false spike in the
sample. Log-odds remains available with
``--transform logodds`` for the country-panel comparison.

The published daily series is the unsmoothed normalized flow, one row per
calendar day, and every day in the sample counts. Smoothing was tested and
rejected; see the SMOOTH_HALFLIFE comment for the numbers.

The precision roster excludes composite "Nothing Ever Happens" parlays.
The audit's only accepted false positive is one of these mixed-outcome
markets, and the family contributes noisy revisions unrelated to the
newspaper count. ``--include-composites`` restores the unfiltered roster.

Horizon is not adjusted for. The claim used to be that a flow framing makes
it self-correct, because the same news moves a one-week market 40 points and
a one-year market 5. That is not what the data shows: 8-30 days carries the
signal on every criterion and beyond 120 days the revision is 13.7% of the
total with a negative GPRD correlation. The reason to leave it alone is
different. Horizon is confounded with subject -- Korea and Taiwan trade
almost entirely long-dated because their risks are structural, where Israeli
strike markets are short-dated because the risk is acute -- so a horizon
filter selects on the kind of risk and deletes whole countries from the
cross-section.

Usage:
    python analysis/index_pm.py --diagnose     # run this FIRST
    python analysis/index_pm.py
    python analysis/index_pm.py --smooth 6 --include-composites
"""

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The raw catalog is the source of record, but the index needs exactly two of
# its columns: neg_risk, which picks the per-event aggregation, and title, for
# --diagnose. events_slim.db carries both without the payload, so a machine
# that only runs the analysis does not need the multi-gigabyte catalog.
SRC_CANDIDATES = ("data/Phase1/Polymarket/polymarket_raw.db",
                  "data/Phase1/Polymarket/events_slim.db")


def _has_events_table(path: str) -> bool:
    """Return whether *path* is a readable catalog with event metadata."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            return con.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name='events'").fetchone() is not None
    except sqlite3.Error:
        return False


_src_paths = [os.path.join(REPO, p) for p in SRC_CANDIDATES]
SRC_DB = next((p for p in _src_paths if _has_events_table(p)), None)
if SRC_DB is None:
    raise FileNotFoundError(
        "no readable catalog with an events table; expected one of "
        + ", ".join(_src_paths))
JUDGE_DB = os.path.join(REPO, "data/Phase2/event_judge.db")
BARS_DB = os.path.join(REPO, "data/Phase3/Polymarket/judged_ohlc.db")
CI_CSV = os.path.join(REPO, "data/CI/gpr_daily.csv")

TRAIL = 90            # trailing window for the live-event normalizer
# Off by default, and the reason is a measurement rather than taste. GPRD's
# own lag-1 autocorrelation is 0.622 and the unsmoothed index already sits at
# 0.658, so the revision flow is as persistent as the target without help. A
# causal EWMA drives the index to 0.989 -- far past the target -- and that
# excess persistence, not new information, is what lifts the level
# correlation: r at lag +2 goes 0.431 -> 0.432 while the correlation of daily
# changes falls 0.177 -> 0.084. Smoothing also moves the peak from +2, where
# the market leads the newspaper, to -1. The knob stays for inspection, but
# the published series is not smoothed.
SMOOTH_HALFLIFE = None
# There is deliberately no minimum-roster gate on the sample. A threshold
# would be a choice that has to be defended on every rerun, and the thin
# months are part of the record: excluding them lifts the headline
# correlation by 0.060 (0.428 -> 0.488 at lag +2) but shortens the published
# window and makes results incomparable with anything measured before it.
# The full two years count. The roster diagnostic below still marks STARVED
# months, so a reader can see where the roster is thin without the index
# silently dropping those days.
# Composite "Nothing Ever Happens" parlays mix geopolitical and unrelated
# outcomes. The audit's only observed accepted false positive is one of these
# markets, so exclude the family from the precision roster by default.
COMPOSITE_MARKER = "nothing ever happens"
EXCLUDE_COMPOSITES = True
# Which normalizer divides the daily sum. "trail" is the trailing mean count
# of live events; "sqrt_live" is the contemporaneous sqrt(n_live)
# alternative. It has real advantages --
# causal on day t, no warm-up window, immune to --start truncation -- but
# the growth stays in: it correlates 0.581 with time against 0.227 for the
# trailing mean, and the regional panel falls from 0.267 to 0.207. The
# premise is that R_sum scales as sqrt(n), which holds for sums of mean-zero
# terms; R_sum sums absolute revisions, and its measured exponent against
# n_live is 1.301 (se 0.029), rejecting 0.5 at z = +27.6.
NORM = "trail"
# Confidence weighting, folded into the primary after the knob sweep. The
# judge writes "high" or "medium"; --no-conf turns it off for comparison.
CONF_WEIGHT = True
CONF_WEIGHTS = {"high": 1.0, "medium": 0.5}
# How a day's revision is measured. The default cap credits a corroborated
# intraday excursion only up to twice the close-to-close move. This keeps
# genuine movement while suppressing the long round trips that are false
# spikes: the worst 2024-09-30 contributor sat between 0.5 and 0.9995 for
# hours and closed at 0.071, so a path that ends where it began earns almost
# nothing however far it travelled.
#
# 1.0 is the value f977290 committed. It was raised to 2.0 during the
# refinement trial and is back: sweeping 1.0 -> 4.0 moves the level
# correlation by +0.003 while costing -0.007 on the correlation of daily
# changes, so a larger cap mostly re-buys persistence rather than adding
# information. 2.0 stays selectable with --wick-cap.
MEASURE = "wick_cap"
WICK_ALPHA = 1.0
WICK_CAP = 1.0
# How several markets on one event combine. "sqrt" divides the summed
# absolute change by sqrt(n) for non-exclusive events; "hybrid" is the
# previous rule, which used the mean. See revisions().
AGG = "sqrt"
# Work in probability by default: it wins on aggregate GPRD correlation.
# --transform logodds remains available for country-panel experiments.
TRANSFORM = "raw"
LOGIT_CLIP = 0.01
THREAT_CATS = (1, 2, 3, 4, 5)
ACT_CATS = (6, 7, 8)


def composite_ids(meta: pd.DataFrame) -> set:
    """Event ids belonging to the mixed composite/parlay family."""
    return set(meta.loc[
        meta["title"].fillna("").str.contains(
            COMPOSITE_MARKER, case=False, regex=False), "event_id"])


def roster_report() -> dict:
    """What the gates keep and what they remove, for labelling any count.

    The judged roster, the priced roster and the published roster are three
    different numbers, and the composite gate is why. Printing "1,479
    events" beside a trial that says "1,500" looks like data loss unless the
    removed family is named with it, so every caller that reports a roster
    size reports this alongside.
    """
    with sqlite3.connect(f"file:{JUDGE_DB}?mode=ro", uri=True) as con:
        judged = int(pd.read_sql(
            "SELECT COUNT(DISTINCT event_id) AS n FROM judgments "
            "WHERE relevant = 1", con)["n"].iloc[0])
    with sqlite3.connect(f"file:{BARS_DB}?mode=ro", uri=True) as con:
        bars = pd.read_sql("SELECT event_id, market_id FROM bars", con)
    with sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True) as con:
        meta = pd.read_sql("SELECT event_id, title FROM events", con)
    for frame in (bars, meta):
        frame["event_id"] = frame["event_id"].astype(str)
    comps = composite_ids(meta)
    bars["composite"] = bars["event_id"].isin(comps)
    kept = bars[~bars["composite"]]
    return dict(judged=judged,
                priced_events=int(bars["event_id"].nunique()),
                priced_markets=int(bars["market_id"].nunique()),
                unpriced=judged - int(bars["event_id"].nunique()),
                composite_events=int(bars.loc[bars["composite"], "event_id"]
                                     .nunique()),
                composite_markets=int(bars.loc[bars["composite"], "market_id"]
                                      .nunique()),
                kept_events=int(kept["event_id"].nunique()),
                kept_markets=int(kept["market_id"].nunique()))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bars and judgments, applying the precision roster gate."""
    for path in (JUDGE_DB, BARS_DB):
        if not os.path.exists(path):
            sys.exit(f"missing {path}")
    with sqlite3.connect(f"file:{JUDGE_DB}?mode=ro", uri=True) as con:
        j = pd.read_sql(
            "SELECT event_id, category, direction, region, confidence "
            "FROM judgments WHERE relevant = 1", con)
    # open/high/low are needed for the default capped intraday excursion:
    # a market that runs 0.10 to 0.30 and back closes unchanged, but the cap
    # credits only a bounded part of that path. n_points remains the only
    # per-bar activity measure in the data now that Polymarket supplies no
    # bar volume.
    with sqlite3.connect(f"file:{BARS_DB}?mode=ro", uri=True) as con:
        b = pd.read_sql(
            "SELECT market_id, event_id, date, open, high, low, close, n_points "
            "FROM bars", con)
    with sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True) as con:
        meta = pd.read_sql("SELECT event_id, neg_risk, title FROM events", con)
    for frame in (j, b, meta):
        frame["event_id"] = frame["event_id"].astype(str)
    if EXCLUDE_COMPOSITES:
        excluded = composite_ids(meta)
        j = j[~j["event_id"].isin(excluded)].copy()
    neg = meta[["event_id", "neg_risk"]]
    b["date"] = pd.to_datetime(b["date"])
    b = b.merge(j, on="event_id").merge(neg, on="event_id", how="left")
    b["neg_risk"] = b["neg_risk"].fillna(0).astype(int)
    return b, j


def revisions(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per event-day: how much probability moved, signed and unsigned.

    A market is live on day t when it has a close on both t-1 and t, so the
    first observed day contributes nothing and the last one contributes its
    full jump.

    The per-event aggregation depends on the event's shape, because the same
    news must not be counted several times:

      negRisk events are a mutually exclusive partition, so probability moving
      from one outcome to another is reallocation, not new risk. Half the sum
      of absolute changes is the mass reallocated, and it reduces to the plain
      binary formula when there are two outcomes.

      Non-exclusive events are a mixed population, not the nested deadlines
      the mean rule assumed. Many are partitions over different subjects --
      one market per country, or per target -- where a story that moves a
      few of nineteen markets should not be divided by nineteen. Dividing by
      sqrt(n) is the middle between summing partially correlated signals and
      averaging perfectly redundant ones.
    """
    bars = bars.sort_values(["market_id", "date"])
    if TRANSFORM == "logodds":
        for c in ("close", "high", "low"):
            q = bars[c].clip(LOGIT_CLIP, 1 - LOGIT_CLIP)
            bars[c] = np.log(q / (1 - q))
    g = bars.groupby("market_id", sort=False)
    bars["dp"] = g["close"].diff()
    bars["gap"] = g["date"].diff().dt.days
    # Keep yesterday's close before dropping first bars and gap resumes.
    # Re-grouping after that filter would make each market's first retained
    # bar look like an unobserved predecessor and silently discard its
    # intraday range.
    bars["prev_close"] = g["close"].shift(1)
    # Last observed day for each market, taken before the gap filter so a
    # market whose final bar follows a gap is still recognized as resolved.
    bars["last_date"] = g["date"].transform("max")
    # A resumed series after a gap is not a one-day revision. Drop those
    # rather than crediting a week of drift to a single day.
    #
    # Measured: this is a safety net, not an active filter. It removes 4 of
    # the 130,126 collected bars because the Phase 3 collector is contiguous.
    # The 4,177 first-bars contribute nothing by construction. Keeping
    # prev_close before this filter also makes the first retained bar's
    # intraday range correct.
    bars = bars[(bars["gap"] == 1) & bars["dp"].notna()].copy()

    # Close-to-close is the day's net revision. A market that runs 0.10 to
    # 0.30 and back closes unchanged, while true_range sees the excursion.
    # The production wick_cap branch retains only a bounded twice-net span,
    # so the path contributes without letting a round trip dominate.
    #
    if MEASURE == "true_range":
        prev = bars["prev_close"]
        hi = np.maximum(bars["high"], prev)
        lo = np.minimum(bars["low"], prev)
        bars["adp"] = (hi - lo).abs().fillna(bars["dp"].abs())
    elif MEASURE == "wick":
        prev = bars["prev_close"]
        hi = np.maximum(bars["high"], prev)
        lo = np.minimum(bars["low"], prev)
        rng = (hi - lo).abs().fillna(bars["dp"].abs())
        bars["adp"] = ((1.0 - WICK_ALPHA) * bars["dp"].abs()
                       + WICK_ALPHA * rng)
    elif MEASURE == "wick_cap":
        prev = bars["prev_close"]
        hi = np.maximum(bars["high"], prev)
        lo = np.minimum(bars["low"], prev)
        rng = (hi - lo).abs().fillna(bars["dp"].abs())
        bars["adp"] = np.minimum(rng, (1.0 + WICK_CAP) * bars["dp"].abs())
    else:
        bars["adp"] = bars["dp"].abs()
    # A market's final bar is its resolution jump. This is the mechanical
    # act/threat split the module docstring describes -- acts are resolutions,
    # threats are revisions in markets still open -- as opposed to the split
    # by judge category, which asks what an event is about rather than what
    # happened to it. Both are built; the 2x2 decides which one the mapping
    # to CI's GPRA and GPRT actually supports.
    bars["adp_term"] = bars["adp"].where(
        bars["date"].eq(bars["last_date"]), 0.0)
    per = bars.groupby(["event_id", "date"]).agg(
        adp_sum=("adp", "sum"), adp_mean=("adp", "mean"),
        adp_term=("adp_term", "sum"),
        dp_sum=("dp", "sum"), dp_mean=("dp", "mean"),
        neg=("neg_risk", "max"), n_mkt=("market_id", "nunique")).reset_index()
    # negRisk partitions are mutually exclusive, so probability moving from
    # one outcome to another is reallocation and half the summed absolute
    # change is the mass that moved.
    #
    # Everything else used to take the mean, justified as nested deadlines
    # -- strike by June, by July, by August -- where one story moves every
    # market and a mean avoids counting it three times. Inspection does not
    # support that. Of the 126 non-exclusive events carrying four or more
    # markets, many are partitions over different SUBJECTS rather than
    # dates: "Which countries will send warships through the Strait of
    # Hormuz" is 19 markets, one per country, and "What will the U.S. or
    # Israel target in Iran" is five different targets. News that moves five
    # of nineteen country-markets is divided by nineteen under a mean.
    #
    # The population is mixed -- some genuine subject partitions, some date
    # partitions that are exclusive in substance but carry no negRisk flag
    # -- so neither the sum nor the mean is right for all of it. Dividing by
    # sqrt(n) is the standard middle: it is what the aggregate of n
    # partially correlated signals scales as, against n for independent ones
    # and 1 for perfectly redundant ones.
    #
    # The evidence is suggestive rather than conclusive, and is recorded as
    # such. Paired over 300 identical event splits it gains +0.029 of
    # split-half reliability with a 95% interval of [-0.005, +0.062], better
    # on 95.3% of splits; the regional panel gains +0.009 with an interval
    # covering zero. All three yardsticks move up and none of them clears
    # significance on its own. It is adopted because the stated reason for
    # the mean does not survive reading the events, not because the numbers
    # settled it. --agg mean restores the previous rule.
    if AGG == "hybrid":
        per["R"] = np.where(per["neg"] == 1, 0.5 * per["adp_sum"],
                            per["adp_mean"])
    else:
        per["R"] = np.where(
            per["neg"] == 1, 0.5 * per["adp_sum"],
            per["adp_sum"] / np.sqrt(per["n_mkt"].clip(lower=1)))
    per["S_raw"] = np.where(per["neg"] == 1, per["dp_sum"], per["dp_mean"])
    # Split R proportionally rather than re-aggregating each side, so both
    # halves inherit the negRisk and nested-deadline handling above and sum
    # back to R exactly.
    share = np.where(per["adp_sum"] > 0, per["adp_term"] / per["adp_sum"], 0.0)
    per["R_resolve"] = per["R"] * share
    per["R_open"] = per["R"] * (1.0 - share)
    return per


def assemble(per: pd.DataFrame, judg: pd.DataFrame,
             days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily aggregates, normalized by the trailing count of live events.

    The normalizer is a trailing mean rather than the same day's count on
    purpose. Polymarket lists markets in response to news, so a shock is
    followed by a burst of listings; dividing by the contemporaneous count
    would mechanically depress the index on exactly the days it should spike.
    A trailing window still removes the secular growth of the universe, which
    is the thing that must go.

    NORM="sqrt_live" selects the contemporaneous alternative,
    R_sum / sqrt(n_live), contemporaneous. Its advantages are real: causal on
    day t, no warm-up, and --start cannot truncate it. Its premise is that
    R_sum scales as sqrt(n_live), which holds for sums of MEAN-ZERO terms
    where the standard deviation grows as sqrt(n). R_sum sums ABSOLUTE
    revisions, every term positive, so its mean grows with n. Regressing
    log(R_sum) on log(n_live) over the 566 published days gives an exponent of
    1.523 (se 0.062), rejecting 0.5 at z = +16.6 and even 1.0 at z = +8.5,
    since on busy days more events are live AND each revises more.

    The published default applies no smoothing. ``--smooth HALFLIFE`` adds a
    causal EWMA to the normalized columns; it is off because it inflates
    level correlation by pushing persistence past the target's own.

    Every day is returned, on the full daily calendar, so a downstream
    ``.shift(lag)`` means lag calendar days rather than lag retained rows.
    """
    per = per.merge(judg[["event_id", "category", "direction", "confidence"]],
                    on="event_id", how="left")
    per["sign"] = per["direction"].map({"up": 1.0, "down": -1.0}).fillna(0.0)

    # The judge recorded how sure it was that an event is geopolitical, and
    # the index ignored it. Down-weighting the medium-confidence events
    # raises both non-headline yardsticks -- the pooled regional correlation
    # from 0.208 to 0.219 and split-half reliability from 0.681 to 0.685 --
    # at no cost to GPRD, and it introduces no threshold to defend, since
    # the weights attach to a label the judge already wrote.
    if CONF_WEIGHT:
        w = per["confidence"].map(CONF_WEIGHTS).fillna(
            np.mean(list(CONF_WEIGHTS.values())))
        for col in ("R", "S_raw", "R_resolve", "R_open"):
            if col in per:
                per[col] = per[col] * w
    per["S"] = per["S_raw"] * per["sign"]

    live = per.groupby("date")["event_id"].nunique().reindex(
        days, fill_value=0).astype(float)
    if NORM == "sqrt_live":
        trail = np.sqrt(live).replace(0, np.nan)
    else:
        trail = live.rolling(TRAIL, min_periods=20).mean().replace(0, np.nan)

    out = pd.DataFrame(index=days)
    out["n_live"] = live
    out["R_sum"] = per.groupby("date")["R"].sum().reindex(days, fill_value=0.0)
    out["S_sum"] = per.groupby("date")["S"].sum().reindex(days, fill_value=0.0)
    for name, cats in (("threat", THREAT_CATS), ("act", ACT_CATS)):
        sub = per[per["category"].isin(cats)]
        out[f"R_{name}"] = sub.groupby("date")["R"].sum().reindex(
            days, fill_value=0.0)
    for col in ("R_resolve", "R_open"):
        out[col] = per.groupby("date")[col].sum().reindex(days, fill_value=0.0)
    out["n_moving"] = per[per["R"] > 0].groupby("date")["event_id"].nunique(
        ).reindex(days, fill_value=0)

    for src_col, dst in (("R_sum", "IDX_sum"), ("S_sum", "IDX_signed"),
                         ("R_threat", "IDX_threat"), ("R_act", "IDX_act"),
                         ("R_resolve", "IDX_resolve"), ("R_open", "IDX_open")):
        out[dst] = out[src_col] / trail
    if SMOOTH_HALFLIFE:
        idx_cols = [c for c in out.columns if c.startswith("IDX_")]
        out[idx_cols] = out[idx_cols].ewm(
            halflife=SMOOTH_HALFLIFE, adjust=True).mean()

    # Returned on the full daily calendar rather than compacted. A hole in the
    # series is not cosmetic: every downstream .shift(lag) would otherwise mean
    # "lag-th retained day" instead of "lag days ago" -- silently wrong lags
    # rather than a visible error. Consumers drop NaN where they need a dense
    # sample.
    return out


def country_gpr() -> pd.DataFrame:
    """CI's monthly country series, wide, indexed by month start.

    These are the GPRC_XXX columns of the monthly export. The coverage is the
    binding constraint on anything regional: CI built country indices for
    countries with financial markets, so the list is large economies and not
    conflict theatres. Iran, Palestine, Yemen, Lebanon, North Korea and Syria
    have no series, and between them they carry more of this roster than
    every matched country combined.
    """
    d = pd.read_excel(os.path.join(REPO, "data/CI/data_gpr_export.xls"))
    cols = [c for c in d.columns if c.upper().startswith("GPRC_")]
    out = d[["month"] + cols].copy()
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M")
    out = out.set_index("month")
    out.columns = [c.replace("GPRC_", "") for c in cols]
    return out


def regional(per: pd.DataFrame, judg: pd.DataFrame, days: pd.DatetimeIndex,
             min_events: int = 10) -> None:
    """Does each region's index track its own country's GPR?

    One aggregate correlation is a single draw. If the index measures
    geopolitical risk rather than market activity, then the Ukraine markets
    should track Ukraine's GPR more closely than Israel's, and the reverse.
    That is a cross-sectional test with as many comparisons as there are
    matched countries, and it is much harder to pass by chance.

    Country GPR is monthly, so the daily index is averaged within month. With
    about two years of overlap each cell rests on ~25 points, which is thin
    for any single country and is why the pooled panel below carries more
    weight than any individual row.
    """
    reg = judg[["event_id", "region"]].dropna()
    cg = country_gpr()
    counts = reg.groupby("region")["event_id"].nunique()
    keep = [r for r in counts[counts >= min_events].index if r in cg.columns]
    keep.sort(key=lambda r: -counts[r])
    matched = int(counts[keep].sum())
    total = int(counts.sum())
    print(f"\n=== regional decomposition, {len(keep)} countries with "
          f">={min_events} events")
    print(f"  matched {matched:,} of {total:,} events ({100*matched/total:.0f}%)"
          f"; the rest are regions CI publishes no country series for")
    if len(keep) < 2:
        print("  too few matched countries to test")
        return

    # One index per region, each with its own trailing normalizer, so a
    # small region is not scaled by the whole roster's activity.
    series = {}
    for r in keep:
        ids = set(reg.loc[reg["region"] == r, "event_id"])
        sub = per[per["event_id"].isin(ids)]
        idx = assemble(sub, judg, days)
        m = idx["IDX_sum"].groupby(idx.index.to_period("M")).mean()
        series[r] = m
    idx_m = pd.DataFrame(series)
    common = idx_m.index.intersection(cg.index)
    idx_m, cg = idx_m.loc[common], cg.loc[common, keep]
    print(f"  {len(common)} overlapping months, "
          f"{common.min()} to {common.max()}")

    print(f"\n  {'region':<8}{'n_ev':>6}{'own':>8}{'others':>8}"
          f"{'rank':>7}{'  own country GPR'}")
    wins, ranks = 0, []
    for r in keep:
        rs = {c: idx_m[r].corr(cg[c]) for c in keep}
        own = rs[r]
        others = [v for c, v in rs.items() if c != r and np.isfinite(v)]
        if not np.isfinite(own) or not others:
            continue
        order = sorted((v for v in rs.values() if np.isfinite(v)),
                       reverse=True)
        rank = order.index(own) + 1
        ranks.append(rank)
        wins += own > max(others)
        print(f"  {r:<8}{counts[r]:>6}{own:>8.3f}{np.mean(others):>8.3f}"
              f"{rank:>4}/{len(order):<3}{'  <-- best' if rank == 1 else ''}")

    if ranks:
        exp = (len(keep) + 1) / 2
        print(f"\n  own country ranks first for {wins} of {len(ranks)} regions"
              f"; mean rank {np.mean(ranks):.1f} against {exp:.1f} by chance")
        # The ranks come from one correlation matrix and are not independent,
        # so the normal approximation would overstate the evidence. Permuting
        # which country counts as "own" reuses the same matrix and holds its
        # correlation structure fixed, which is the null this test needs.
        M = np.array([[idx_m[r].corr(cg[c]) for c in keep] for r in keep])
        rng = np.random.default_rng(20260908)
        obs = float(np.mean(ranks))
        null = np.empty(10000)
        for k in range(null.size):
            pi = rng.permutation(len(keep))
            null[k] = np.mean([
                1 + int((np.sort(M[i][np.isfinite(M[i])])[::-1]
                         > M[i, pi[i]]).sum())
                for i in range(len(keep)) if np.isfinite(M[i, pi[i]])])
        p = float((null <= obs).mean())
        print(f"  permutation test, 10,000 relabelings: p = {p:.4f} "
              f"(null mean rank {null.mean():.2f})")

    # Pooled panel. Standardizing within country removes the level and scale
    # differences between countries, so what is left is whether a country's
    # busy months on Polymarket are its busy months in the newspapers.
    z_i = (idx_m - idx_m.mean()) / idx_m.std()
    z_c = (cg - cg.mean()) / cg.std()
    long = pd.DataFrame({"i": z_i.stack(), "c": z_c.stack()}).dropna()
    if len(long) > 10:
        within = long["i"].corr(long["c"])
        # Demeaning by month as well removes anything common to all
        # countries in a month, which is the global news cycle.
        d2 = long.copy()
        for col in ("i", "c"):
            # level 0 is the month. zi.stack() yields (month, region), and
            # the columns were already standardized within region, so
            # demeaning by level 1 subtracts a mean that is already zero and
            # silently does nothing.
            d2[col] -= d2.groupby(level=0)[col].transform("mean")
        print(f"  pooled within-country r = {within:+.3f} "
              f"on {len(long):,} country-months")
        print(f"  after removing month effects r = "
              f"{d2['i'].corr(d2['c']):+.3f}")


CAT_NAMES = {1: "war threats", 2: "peace threats", 3: "military buildup",
             4: "nuclear threats", 5: "terror threats", 6: "war begins",
             7: "war escalation", 8: "terror acts"}


def categories(per: pd.DataFrame, judg: pd.DataFrame,
               days: pd.DatetimeIndex, min_events: int = 20) -> None:
    """Does the market's category mix track the newspapers' category mix?

    The judge assigns every event one of CI's own eight categories, and CI
    publish SHAREH_CAT_1..8 monthly -- the share of GPR articles falling in
    each. So the two sides can be compared as compositions: in a month when
    the newspapers are unusually about nuclear threats, is the market too?

    This is the threat/act 2x2 at eight-way resolution instead of two, and
    it is the same shape of test as the regional decomposition on a
    different axis. Shares rather than levels on both sides, because a
    composition removes the common size effect for free -- a busy month
    lifts every category on both sides and would otherwise manufacture the
    correlation this is trying to find.
    """
    d = pd.read_excel(os.path.join(REPO, "data/CI/data_gpr_export.xls"))
    d["month"] = pd.to_datetime(d["month"]).dt.to_period("M")
    ci = d.set_index("month")[[f"SHAREH_CAT_{i}" for i in range(1, 9)]]
    ci.columns = list(range(1, 9))

    p = per.merge(judg[["event_id", "category"]], on="event_id", how="left")
    p = p.dropna(subset=["category"])
    # Respect the requested window. per carries every bar back to 2023-02,
    # and the coverage work established the early months are too thin to
    # treat as comparable to the rest.
    p = p[p["date"].isin(days)]
    p["month"] = p["date"].dt.to_period("M")
    counts = judg.groupby("category")["event_id"].nunique()
    keep = [int(c) for c in counts[counts >= min_events].index
            if int(c) in ci.columns]

    wide = p.pivot_table(index="month", columns="category", values="R",
                         aggfunc="sum", fill_value=0.0)
    wide = wide[[c for c in keep if c in wide.columns]]
    mkt = wide.div(wide.sum(axis=1), axis=0)          # market shares
    common = mkt.index.intersection(ci.index)
    mkt, news = mkt.loc[common], ci.loc[common, list(mkt.columns)]
    news = news.div(news.sum(axis=1), axis=0)         # renormalize to kept

    print(f"\n=== category composition, {len(mkt.columns)} of 8 categories "
          f"with >={min_events} events")
    print(f"  {len(common)} months, {common.min()} to {common.max()}")
    print(f"\n  {'cat':<4}{'name':<18}{'n_ev':>6}{'mkt share':>11}"
          f"{'news share':>12}{'corr':>8}")
    rs = {}
    for c in mkt.columns:
        r = mkt[c].corr(news[c])
        rs[c] = r
        print(f"  {c:<4}{CAT_NAMES.get(c, '?'):<18}{int(counts[c]):>6}"
              f"{mkt[c].mean():>11.1%}{news[c].mean():>12.1%}{r:>8.3f}")

    # Pooled, both sides standardized within category and month effects
    # removed. Same estimator as the regional panel, so the two numbers are
    # directly comparable.
    zi = (mkt - mkt.mean()) / mkt.std()
    zc = (news - news.mean()) / news.std()
    long = pd.DataFrame({"i": zi.stack(), "c": zc.stack()}).dropna()
    if len(long) > 20:
        pooled = long["i"].corr(long["c"])
        d2 = long.copy()
        for col in ("i", "c"):
            d2[col] -= d2.groupby(level=0)[col].transform("mean")
        print(f"\n  pooled within-category r = {pooled:+.3f} "
              f"on {len(long)} category-months")
        print(f"  after removing month effects r = "
              f"{d2['i'].corr(d2['c']):+.3f}")

    # Whether the market's busiest category is the newspapers' busiest one
    # is a cruder question than the correlation, and a reader can check it
    # without trusting any of the standardizing above.
    agree = (mkt.idxmax(axis=1) == news.idxmax(axis=1)).mean()
    print(f"  top category agrees in {agree:.0%} of months "
          f"({1 / len(mkt.columns):.0%} by chance)")


def two_by_two(x: pd.DataFrame, cols: tuple) -> None:
    """Print one threat/act block and say whether its diagonal holds.

    The first column named is the threat side, the second the act side, so
    the diagonal is (first, GPRD_THREAT) and (second, GPRD_ACT). A row is
    marked ok when its own target beats the other one; the mapping is only
    evidence when both rows are ok.
    """
    targets = ("GPRD_THREAT", "GPRD_ACT")
    for i, col in enumerate(cols):
        if col not in x:
            continue
        rs = [x[col].corr(x[t]) for t in targets]
        own, other = rs[i], rs[1 - i]
        mark = "ok" if own > other else f"inverted by {other - own:+.3f}"
        print(f"  {col:<14} {rs[0]:>12.3f} {rs[1]:>10.3f}   {mark}")


def ci_series() -> pd.DataFrame:
    ci = pd.read_csv(CI_CSV, parse_dates=["date"]).set_index("date")
    return ci[["GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7"]]


def periods(x: pd.DataFrame, lag: int = 2) -> None:
    """Does the correlation hold across the sample, or only in episodes?

    The headline is a full-sample average, and a referee's first move is to
    cut it by period. Doing that here gives roughly 0.5 in the two stretches
    containing a major episode and roughly 0 in the two between them.

    Range restriction is the obvious explanation and it is not the answer.
    The quiet stretches do not have much narrower GPRD spread -- 48 to 51
    against 72 for the full sample -- and correcting for attenuation with
    Thorndike's case 2 leaves the dead periods dead.

    Roster size is not the answer either: the deadest stretch carries the
    highest live-event count in the sample.

    Read this next to episodes(), not instead of it, because the boundaries
    do a lot of work. These cuts are placed to ISOLATE the stretches between
    episodes. The six-month blocks in episodes() straddle them instead and
    look far healthier -- +0.09, +0.54, +0.36, +0.39 -- and a block moved by
    one month at each end goes from -0.03 to +0.36, because that month is
    March 2026.

    Both are honest and they answer different questions. Across arbitrary
    blocks the correlation holds up. Between episodes specifically it does
    not, which is the same fact the conditional-on-GPRD-level table reports
    from the other direction: above the median r is +0.545, below it r is
    near zero. The claim to make is that the index tracks GPRD during
    episodes, not that it tracks GPRD on average.
    """
    cuts = [("2024-09-08", "2025-01-31", "thin roster"),
            ("2025-02-01", "2025-08-31", "Iran strike episode"),
            ("2025-09-01", "2026-02-28", "between episodes"),
            ("2026-03-01", "2026-09-01", "March 2026 episode")]
    sd_all = x["GPRD"].std()
    print(f"\n=== by period: is the correlation episodic?")
    print(f"  {'period':<22}{'days':>6}{'live/day':>10}{'sd GPRD':>9}"
          f"{'r':>8}{'attn-adj':>10}")
    for lo, hi, lab in cuts:
        s = x.loc[lo:hi]
        if len(s) < 60:
            continue
        r = s["IDX_sum"].shift(lag).corr(s["GPRD"])
        ratio = sd_all / s["GPRD"].std()
        adj = (r * ratio / np.sqrt(1 + r * r * (ratio ** 2 - 1))
               if np.isfinite(r) else np.nan)
        nlv = s["n_live"].mean() if "n_live" in s else float("nan")
        print(f"  {lab:<22}{len(s):>6}{nlv:>10.0f}{s['GPRD'].std():>9.0f}"
              f"{r:>8.3f}{adj:>10.3f}")
    print("  These cuts isolate the between-episode stretches on purpose.")
    print("  episodes() straddles them and reads +0.09/+0.54/+0.36/+0.39;")
    print("  one month of boundary shift moves a block from -0.03 to +0.36.")
    print("  The defensible claim is that the index tracks GPRD during")
    print("  episodes, not that it tracks GPRD on average.")


def deseason(a: pd.Series, b: pd.Series, fit_end: str = None) -> tuple:
    """Divide the weekly publication cycle out of both series.

    Newspapers have a week; prediction markets do not. GPRD falls to 0.66 of
    its weekday level at the weekend while the index only falls to 0.90, so
    part of every daily correlation between them is a calendar mismatch
    rather than geopolitics. The fix is symmetric -- each series is divided
    by its own day-of-week mean -- so it cannot flatter one side.

    Sized honestly: on the full sample this lifts the peak correlation from
    0.391 to 0.456, but with the factors fit on training data only the
    holdout gain is +0.004 at the peak and +0.047 one lag earlier. Most of
    the full-sample figure is in-sample fit. It is kept because a weekly
    publication cycle is not geopolitical risk and adjusting for it is
    standard for daily series, not because it buys much.

    fit_end restricts factor estimation to a training period.
    """
    dow = a.index.dayofweek
    fit = slice(None) if fit_end is None else slice(None, fit_end)
    out = []
    for s in (a, b):
        f = s.loc[fit].groupby(s.loc[fit].index.dayofweek).mean()
        out.append(s / pd.Series(dow, index=s.index).map(f).replace(0, np.nan))
    return tuple(out)


def xcorr(x: pd.Series, y: pd.Series, lags=range(-10, 11)) -> pd.Series:
    """Cross-correlation. A POSITIVE lag means x leads y by that many days."""
    return pd.Series({k: x.shift(k).corr(y) for k in lags})


def diagnose(per: pd.DataFrame, idx: pd.DataFrame, ci: pd.DataFrame,
             titles: dict) -> None:
    """The concentration test, run before any correlation is believed.

    The likely failure of this whole design is that the many drown the few:
    two thousand thin markets each bouncing a tick produce a baseline that
    swamps the ten markets that actually moved on a news day. The symptom is
    an index whose biggest days are unremarkable Tuesdays.

    So look at the days directly. If the top days are recognizable news and
    the top contributors are the obvious markets, the signal is there and
    normalization is a detail. If they are a diffuse smear of three-cent
    moves, no aggregation choice rescues it.
    """
    print("\n=== dispersion: a real index should be spikier than its median")
    for name, s in (("GPRD", ci["GPRD"]), ("IDX_sum", idx["IDX_sum"]),
                    ("R_sum", idx["R_sum"])):
        s = s.dropna()
        med = s.median() or float("nan")
        print(f"  {name:<9} max/median {s.max() / med:>8.2f}   "
              f"p99/median {s.quantile(.99) / med:>7.2f}   "
              f"median {med:.4f}")

    print("\n=== day of week, as a ratio to the series mean")
    print("    Polymarket trades on weekends and newspapers thin out, so an")
    print("    untreated weekend effect alone can pull a real correlation to 0")
    for name, s in (("GPRD", ci["GPRD"]), ("IDX_sum", idx["IDX_sum"]),
                    ("n_live", idx["n_live"])):
        s = s.dropna()
        by = s.groupby(s.index.dayofweek).mean() / s.mean()
        print(f"  {name:<9} " + "  ".join(
            f"{d}:{by.get(i, float('nan')):.2f}"
            for i, d in enumerate("Mo Tu We Th Fr Sa Su".split())))

    for label, series in (("CI GPRD", ci["GPRD"]),
                          ("our IDX_sum", idx["IDX_sum"])):
        print(f"\n=== top 8 days by {label}, with our biggest contributors")
        top = series.dropna().reindex(idx.index).dropna().nlargest(8)
        for day, val in top.items():
            gp = ci["GPRD"].get(day, float("nan"))
            ours = idx["IDX_sum"].get(day, float("nan"))
            print(f"  {day.date()}  GPRD {gp:>6.0f}  ours {ours:>7.3f}")
            sub = per[per["date"] == day].nlargest(3, "R")
            for _, r in sub.iterrows():
                print(f"      R={r['R']:.3f}  "
                      f"{titles.get(r['event_id'], '?')[:64]}")
    paired = idx[["IDX_sum"]].join(ci[["GPRD"]]).dropna(
        subset=["IDX_sum", "GPRD"])
    # Scoring at lag 0 alone punishes the market's lead. GPRD is a count of
    # articles about a risk, so an event the market prices on day t shows up
    # in print a day or two later; the reported cross-correlogram peaks at
    # +2. Report both, and treat the peak-lag column as the precision claim.
    print("\n=== spike audit: the largest index days vs the news on the "
          "same day and at the lead")
    top = paired["IDX_sum"].nlargest(min(25, len(paired))).index
    best = int(xcorr(paired["IDX_sum"], paired["GPRD"]).idxmax())
    cnt = {k: int((paired["GPRD"].shift(-k).rank(pct=True)
                   .loc[top] >= 0.80).sum()) for k in (0, best)}
    print(f"  n={len(paired):>4}  {cnt[0]}/{len(top)} at lag 0, "
          f"{cnt[best]}/{len(top)} at the peak lag +{best}")


def distributed_lag(x: pd.Series, y: pd.Series, k: int = 10) -> None:
    """Regress GPRD on k lags of the index, with Newey-West errors.

    A contemporaneous correlation forces one answer to a question the data
    should answer: how fast does newspaper attention respond to a belief
    revision, and for how long does it persist? A distributed lag lets the
    coefficients trace that decay instead of a chosen smoothing half-life.

    Newey-West rather than plain OLS errors because both series are strongly
    autocorrelated, and ordinary standard errors on autocorrelated residuals
    are too small by a wide margin.
    """
    import statsmodels.api as sm

    frame = pd.DataFrame({"y": y})
    for lag in range(k + 1):
        frame[f"L{lag}"] = x.shift(lag)
    frame = frame.dropna()
    if len(frame) < 60:
        print("  too few overlapping days for a distributed lag")
        return
    model = sm.OLS(frame["y"], sm.add_constant(frame.drop(columns="y")))
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": k})
    print(f"  n={int(res.nobs)}  R2={res.rsquared:.3f}  "
          f"F(joint lags) p={res.f_pvalue:.2e}")
    print(f"  {'lag':>4} {'beta':>10} {'t':>7} {'p':>9}")
    for lag in range(k + 1):
        name = f"L{lag}"
        star = " *" if res.pvalues[name] < 0.05 else ""
        print(f"  {lag:>4} {res.params[name]:>10.2f} "
              f"{res.tvalues[name]:>7.2f} {res.pvalues[name]:>9.3f}{star}")
    total = sum(res.params[f"L{i}"] for i in range(k + 1))
    print(f"  cumulative effect over {k} days: {total:.1f} GPRD points "
          f"per unit of index")


def country_panel(per: pd.DataFrame, judg: pd.DataFrame) -> None:
    """Country-month panel against CI's per-country GPRC series.

    The aggregate daily comparison has a few hundred observations and one
    time series. The panel has a country dimension as well, and country fixed
    effects absorb every constant difference in how much Polymarket lists
    about a place and how much the ten newspapers write about it. What is
    left is within-country variation over time, which is the identification
    that matters.

    The obvious limitation: CI publish about 44 countries, and Iran -- the
    second largest region in this roster -- is not one of them.
    """
    import statsmodels.api as sm

    monthly = pd.read_excel(os.path.join(REPO, "data/CI/data_gpr_export.xls"))
    monthly["month"] = pd.to_datetime(monthly["month"]).dt.to_period("M")
    gprc = {c[5:]: c for c in monthly.columns if c.startswith("GPRC_")}

    per = per.merge(judg[["event_id", "region"]], on="event_id", how="left")
    per["month"] = per["date"].dt.to_period("M")
    ours = per[per["region"].isin(gprc)].groupby(
        ["region", "month"])["R"].sum().rename("R").reset_index()
    if ours.empty:
        print("  no regions overlap the CI country list"); return

    long = monthly.melt(id_vars="month", value_vars=list(gprc.values()),
                        var_name="col", value_name="gprc")
    long["region"] = long["col"].str[5:]
    panel = ours.merge(long[["month", "region", "gprc"]],
                       on=["month", "region"], how="inner").dropna()
    panel = panel[panel["gprc"] > 0]
    if len(panel) < 40:
        print(f"  only {len(panel)} country-months; too few to regress")
        return

    panel["ly"] = np.log(panel["gprc"])
    panel["lx"] = np.log1p(panel["R"])
    dummies = pd.get_dummies(panel["region"], prefix="c", drop_first=True,
                             dtype=float)
    tdum = pd.get_dummies(panel["month"].astype(str), prefix="t",
                          drop_first=True, dtype=float)
    print(f"  {len(panel):,} country-months, "
          f"{panel.region.nunique()} countries, "
          f"{panel.month.nunique()} months")
    for label, extra in (("country FE", dummies),
                         ("country + month FE", pd.concat([dummies, tdum],
                                                          axis=1))):
        X = sm.add_constant(pd.concat(
            [panel[["lx"]].reset_index(drop=True),
             extra.reset_index(drop=True)], axis=1))
        res = sm.OLS(panel["ly"].reset_index(drop=True), X).fit(
            cov_type="cluster",
            cov_kwds={"groups": panel["region"].reset_index(drop=True)})
        print(f"  {label:<20} beta={res.params['lx']:+.3f}  "
              f"t={res.tvalues['lx']:+.2f}  p={res.pvalues['lx']:.4f}  "
              f"R2={res.rsquared:.3f}")


def holdout(x: pd.DataFrame, split: str | None = None,
            lag: int = 2) -> None:
    """Measure the lead-lag peak on two windows that share no days.

    The nested start-date check is weak evidence. Six windows that all
    contain 2025-04..2026-03 move together if a handful of days carry the
    result, and none of the six can show that. Two disjoint windows can.

    The index is built once, before the split, so both windows use the same
    normalizer and the same design. Only the evaluation period changes. The
    sqrt normalizer is causal, so an early window never sees a later day.

    This is not a pristine holdout. Every design choice was made while
    looking at the whole sample. It is still the strongest available test
    that the +2 peak is a property of the index and not of one stretch of
    news.
    """
    import statsmodels.api as sm

    if split is None:
        split = x.index[len(x) // 2].strftime("%Y-%m-%d")
    windows = {
        "A (early)": x.loc[x.index < split],
        "B (late)": x.loc[x.index >= split],
    }
    print(f"\n=== disjoint holdout, split at {split}")

    for name, w in windows.items():
        if len(w) < 90:
            print(f"  {name}: {len(w)} days, too few to test")
            continue
        cc = xcorr(w["IDX_sum"], w["GPRD"])
        peak = int(cc.idxmax())
        r2 = cc[lag]
        # Fisher z interval on the correlation at the pinned lag.
        n = int(w[["IDX_sum", "GPRD"]].dropna().shape[0]) - abs(lag)
        z = np.arctanh(r2)
        se = 1.0 / np.sqrt(max(n - 3, 1))
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        print(f"\n  {name}  {w.index.min().date()} .. "
              f"{w.index.max().date()}  ({len(w)} days)")
        print(f"    peak lag {peak:+d} (r={cc.max():+.3f})   "
              f"r at pinned lag {lag:+d} = {r2:+.3f} "
              f"[95% {lo:+.3f}, {hi:+.3f}]")
        line = "  ".join(f"{k:+d}:{cc[k]:+.2f}" for k in range(-5, 6))
        print(f"    {line}")

        frame = pd.DataFrame({"y": w["GPRD"],
                              "xl": w["IDX_sum"].shift(lag)}).dropna()
        res = sm.OLS(frame["y"], sm.add_constant(frame[["xl"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 10})
        print(f"    Newey-West beta={res.params['xl']:+.1f}  "
              f"t={res.tvalues['xl']:+.2f}  p={res.pvalues['xl']:.4f}")

    a, b = windows["A (early)"], windows["B (late)"]
    if len(a) >= 90 and len(b) >= 90:
        ra = xcorr(a["IDX_sum"], a["GPRD"])[lag]
        rb = xcorr(b["IDX_sum"], b["GPRD"])[lag]
        na = a[["IDX_sum", "GPRD"]].dropna().shape[0] - abs(lag)
        nb = b[["IDX_sum", "GPRD"]].dropna().shape[0] - abs(lag)
        zdiff = (np.arctanh(ra) - np.arctanh(rb)) / np.sqrt(
            1 / max(na - 3, 1) + 1 / max(nb - 3, 1))
        from scipy import stats
        pd_ = 2 * (1 - stats.norm.cdf(abs(zdiff)))
        print(f"\n  A vs B at lag {lag:+d}: z={zdiff:+.2f}, p={pd_:.3f}")
        print("    a high p here means the two windows agree, "
              "which is the result we want")


def episodes(x: pd.DataFrame, lag: int = 2) -> None:
    """Does the lead depend on how loud the news is, or on our coverage?

    The disjoint holdout shows the +2 peak in some windows and not others.
    Two explanations compete. Either the index only works when a large
    episode is running, or the windows differ in how many events the roster
    carries. This table separates them, because roster density and GPRD
    volatility do not move together across these blocks.
    """
    print(f"\n=== disjoint blocks: signal against news volatility "
          f"and roster density")
    print(f"  {'block':<20}{'n':>5}{'r(+' + str(lag) + ')':>8}"
          f"{'GPRD mean':>11}{'GPRD sd':>9}{'GPRD max':>10}{'live ev/day':>12}")
    edges = pd.date_range(x.index.min(), x.index.max(), freq="2QS")
    edges = list(edges) + [x.index.max() + pd.Timedelta(days=1)]
    for a, b in zip(edges, edges[1:]):
        w = x.loc[(x.index >= a) & (x.index < b)]
        if len(w) < 90:
            continue
        r = xcorr(w["IDX_sum"], w["GPRD"])[lag]
        name = f"{a:%Y-%m}..{(b - pd.Timedelta(days=1)):%Y-%m}"
        print(f"  {name:<20}{len(w):>5}{r:>+8.3f}{w.GPRD.mean():>11.0f}"
              f"{w.GPRD.std():>9.0f}{w.GPRD.max():>10.0f}"
              f"{w.n_live.mean():>12.0f}")

    print(f"\n=== correlation at lag {lag:+d}, conditional on GPRD level")
    xs, g = x["IDX_sum"].shift(lag), x["GPRD"]
    for q, label in ((0.0, "all days"), (.50, "GPRD above median"),
                     (.75, "GPRD above p75"), (.90, "GPRD above p90")):
        sub = pd.concat([xs[g >= g.quantile(q)],
                         g[g >= g.quantile(q)]], axis=1).dropna()
        print(f"  {label:<22} n={len(sub):>4}  "
              f"r={sub.iloc[:, 0].corr(sub.iloc[:, 1]):+.3f}")


def reliability(per: pd.DataFrame, judg: pd.DataFrame,
                days: pd.DatetimeIndex, x: pd.DataFrame,
                lag: int = 2, n_iter: int = 200,
                seed: int = 20260907) -> None:
    """Split-half reliability: the ceiling on every external correlation.

    Partition the event roster at random, build the index separately on each
    half, and correlate the two series. Nothing else in this file can exceed
    the result. An index that does not track a second index of its own design,
    built on disjoint events over the same days, cannot be tracking GPRD
    either -- whatever agreement it shows would be coincidence.

    The split is at the event level because the event is the unit the index
    sums over. Splitting on markets would put the two halves of one negRisk
    partition on opposite sides, and each half would then score reallocation
    that only the other half can explain.

    Each half is normalized by its own trailing live count, so both halves are
    the same estimator applied to a smaller roster rather than one estimator
    cut in two. That is what makes the Spearman-Brown step below legitimate:
    it converts a half-length reliability into the full-length one, and
    reliability rises with length, so the raw half correlation understates the
    index the project actually reports.

    The disattenuated figure divides the observed GPRD correlation by the
    square root of reliability. GPRD carries measurement error of its own that
    is not estimated here, so treating it as perfectly reliable makes the
    printed value a lower bound on the true relationship, not a claim about
    it.
    """
    events = per["event_id"].unique()
    if len(events) < 4:
        print("\n=== split-half reliability: too few events")
        return

    rng = np.random.default_rng(seed)
    rs, n_days, pairs = [], [], []
    for _ in range(n_iter):
        perm = rng.permutation(events)
        cut = len(perm) // 2
        a = assemble(per[per["event_id"].isin(set(perm[:cut]))], judg, days)
        b = assemble(per[per["event_id"].isin(set(perm[cut:]))], judg, days)
        both = a[["IDX_sum"]].join(b[["IDX_sum"]], how="inner",
                                   lsuffix="_a", rsuffix="_b").dropna()
        # Scored on the days the index actually publishes, which is all of them.
        both = both[both.index.isin(x.index)]
        # A draw that leaves too few shared days says nothing about the
        # design; it says the trailing normalizer had nothing to warm up on.
        if len(both) < 60:
            continue
        rs.append(both["IDX_sum_a"].corr(both["IDX_sum_b"]))
        n_days.append(len(both))
        pairs.append(both)

    if not rs:
        print("\n=== split-half reliability: no draw produced enough days")
        return

    r = np.asarray(rs, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        print("\n=== split-half reliability: correlations undefined")
        return

    half = float(r.mean())
    lo, hi = np.percentile(r, [2.5, 97.5])
    print(f"\n=== split-half reliability, {len(r)} draws of "
          f"{len(events):,} events, median {int(np.median(n_days))} days")
    print(f"  half-length r        {half:>+7.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  sd {r.std(ddof=1):.3f}")

    # Spearman-Brown is only meaningful for a positive half correlation; at
    # r <= 0 the halves share no signal and there is nothing to step up.
    if half <= 0:
        print("  the halves do not agree: the index is noise at this roster "
              "size, and no external correlation it shows is interpretable")
        return

    full = 2 * half / (1 + half)
    print(f"  Spearman-Brown       {full:>+7.3f}   reliability of the full "
          f"roster")

    obs = x["IDX_sum"].shift(lag).corr(x["GPRD"])
    if not np.isfinite(obs):
        return
    print(f"\n  observed r vs GPRD at lag {lag:+d}   {obs:>+7.3f}")

    # Dividing by a near-zero reliability explodes, and the quotient can
    # exceed 1, which is not a correlation. Two guards, in the order the
    # failures actually arrive.
    if lo <= 0:
        print("  disattenuation withheld: reliability is not distinguishable "
              "from zero, so the correction has no stable denominator")
        return

    dis = obs / np.sqrt(full)
    if abs(dis) > 1:
        print(f"  disattenuation withheld: the correction returns "
              f"{dis:+.3f}, outside [-1, 1]. Reliability ({full:+.3f}) is too "
              f"low to divide by -- report the observed r and the ceiling "
              f"separately")
        return

    print(f"  disattenuated (lower bound)   {dis:>+7.3f}")
    print(f"  share of the ceiling used     {abs(obs) / np.sqrt(full):>7.1%}")
    regimes(pairs, x, lag)


def regimes(pairs: list, x: pd.DataFrame, lag: int) -> None:
    """Does the index get more reliable when there is more news, or just more
    correlated?

    The draft reports the GPRD correlation rising with news volume and reads
    it as quiet days having little to revise. That explanation is a claim
    about measurement: if the index is simply noisier when nothing happens,
    reliability should rise with GPRD alongside the correlation, and the two
    facts are one fact.

    The alternative is that reliability is flat and only the correlation
    moves, which would mean the relationship itself strengthens with news
    rather than the measurement improving. That is the more interesting
    claim and it needs the reliability column to support it.

    Both halves are shifted by the same lag as the headline correlation, so
    the reliability is measured on the very days that feed it and the ceiling
    comparison is like for like.

    One confound is unavoidable and is printed rather than hidden. Selecting
    days by GPRD restricts the range of GPRD inside each row, and a restricted
    range attenuates a correlation on its own. The `sd GPRD` column is the
    size of that effect: a row whose spread is much narrower than the full
    sample's cannot be compared to it on `obs r` alone. Reliability is a
    correlation between two halves of the index, so it is not restricted by
    the same selection, only by whatever the selection does to the index.
    """
    g = x["GPRD"]
    cuts = [("all days", 0.0, False), ("GPRD below median", .50, True),
            ("GPRD above median", .50, False), ("GPRD above p75", .75, False),
            ("GPRD above p90", .90, False)]

    print(f"\n=== reliability by news regime, both halves shifted {lag:+d}")
    print(f"  {'regime':<20}{'days':>6}{'half-r':>9}{'SB rel':>9}"
          f"{'ceiling':>9}{'obs r':>8}{'% ceil':>9}{'sd GPRD':>10}")
    for label, q, below in cuts:
        thr = g.quantile(q)
        mask = (g < thr) if below else (g >= thr)
        days_in = x.index[mask.reindex(x.index, fill_value=False)]
        if len(days_in) < 60:
            continue

        rr = []
        for both in pairs:
            sub = both.shift(lag).reindex(days_in).dropna()
            if len(sub) >= 30:
                rr.append(sub["IDX_sum_a"].corr(sub["IDX_sum_b"]))
        rr = np.asarray([v for v in rr if np.isfinite(v)], dtype=float)
        if rr.size == 0:
            continue

        half = float(rr.mean())
        obs = x["IDX_sum"].shift(lag).reindex(days_in).corr(
            g.reindex(days_in))
        row = (f"  {label:<20}{len(days_in):>6}{half:>+9.3f}")
        if half <= 0:
            print(row + f"{'--':>9}{'--':>9}{obs:>+8.3f}{'--':>9}"
                        f"{g.reindex(days_in).std():>10.0f}")
            continue
        sb = 2 * half / (1 + half)
        ceil = np.sqrt(sb)
        share = abs(obs) / ceil if np.isfinite(obs) and ceil > 0 else np.nan
        print(row + f"{sb:>+9.3f}{ceil:>+9.3f}{obs:>+8.3f}"
                    f"{share:>8.1%}{g.reindex(days_in).std():>10.0f}"
                    + ("" if share <= 1 else "  >1"))


def main() -> None:
    global CONF_WEIGHT, MEASURE, AGG, NORM, TRANSFORM, WICK_ALPHA
    global WICK_CAP, SMOOTH_HALFLIFE, EXCLUDE_COMPOSITES
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", default="2024-08-01")
    ap.add_argument("--end", default="2026-09-01")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="lead-lag measured on two windows sharing no days")
    ap.add_argument("--split", default=None,
                    help="YYYY-MM-DD boundary for --holdout (default: median day)")
    ap.add_argument("--regress", action="store_true",
                    help="distributed-lag regression and the country panel")
    ap.add_argument("--reliability", action="store_true",
                    help="split-half reliability: the ceiling on any "
                         "external correlation")
    ap.add_argument("--transform", default=TRANSFORM,
                    choices=("raw", "logodds"),
                    help="probability space, or log-odds")
    ap.add_argument("--smooth", type=float, default=SMOOTH_HALFLIFE or 0.0,
                    help="causal EWMA half-life in days; 0 (default) "
                         "disables it, which is the measured best setting")
    ap.add_argument("--include-composites", action="store_true",
                    help="retain 'Nothing Ever Happens' composite parlays")
    ap.add_argument("--norm", default=NORM, choices=("trail", "sqrt_live"),
                    help="normalizer: trailing mean count, or contemporaneous "
                         "sqrt(n_live)")
    ap.add_argument("--measure", default=MEASURE,
                    choices=("true_range", "close", "wick", "wick_cap"),
                    help="how a day's revision is measured")
    ap.add_argument("--wick-alpha", type=float, default=WICK_ALPHA,
                    help="with --measure wick: share of the intraday "
                         "excursion credited as revision (1 = true_range, "
                         "0 = close)")
    ap.add_argument("--wick-cap", type=float, default=WICK_CAP,
                    help="with --measure wick_cap: credit the excursion only "
                         "up to (1 + wick_cap) x the close-to-close move")
    ap.add_argument("--agg", default=AGG, choices=("sqrt", "hybrid"),
                    help="how several markets on one event combine")
    ap.add_argument("--no-conf", action="store_true",
                    help="disable confidence weighting (pre-sweep primary)")
    ap.add_argument("--categories", action="store_true",
                    help="category composition against CI's SHAREH_CAT_*")
    ap.add_argument("--regional", action="store_true",
                    help="per-region index against CI's country GPR")
    ap.add_argument("--min-events", type=int, default=10,
                    help="minimum events for a region to get its own index")
    ap.add_argument("--iters", type=int, default=200,
                    help="draws for --reliability (default 200)")
    args = ap.parse_args()

    MEASURE = args.measure
    WICK_ALPHA = args.wick_alpha
    WICK_CAP = args.wick_cap
    AGG = args.agg
    NORM = args.norm
    TRANSFORM = args.transform
    SMOOTH_HALFLIFE = args.smooth if args.smooth > 0 else None
    EXCLUDE_COMPOSITES = not args.include_composites
    if args.no_conf:
        CONF_WEIGHT = False

    bars, judg = load()
    print(f"configuration: transform={TRANSFORM}, measure={MEASURE}, "
          f"wick_cap={WICK_CAP:g}, smooth={SMOOTH_HALFLIFE or 0:g}, "
          f"composites={'included' if not EXCLUDE_COMPOSITES else 'excluded'}")
    print(f"{len(bars):,} bar-days, {bars.event_id.nunique():,} events, "
          f"{bars.market_id.nunique():,} markets")
    # Name what the gates removed with the count, so this number can be
    # compared against a pre-gate run without looking like data loss.
    rep = roster_report()
    print(f"roster: {rep['kept_events']:,} of {rep['priced_events']:,} priced "
          f"events, {rep['kept_markets']:,} of {rep['priced_markets']:,} "
          f"markets; {rep['composite_events']} composite parlays "
          f"({rep['composite_markets']} markets) "
          f"{'excluded' if EXCLUDE_COMPOSITES else 'included'}; "
          f"{rep['unpriced']} of {rep['judged']:,} judged events have no bars")
    per = revisions(bars)
    days = pd.date_range(args.start, args.end, freq="D")
    idx = assemble(per, judg, days)
    ci = ci_series().reindex(days)

    src = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
    titles = {str(a): b for a, b in src.execute(
        "SELECT event_id, title FROM events")}
    src.close()

    if args.diagnose:
        diagnose(per, idx, ci, titles)
        return

    if args.categories:
        categories(per, judg, days)
        return

    if args.regional:
        regional(per, judg, days, min_events=args.min_events)
        return

    # A thin roster is the failure mode behind several observed artifacts,
    # so surfacing it per month is worth keeping regardless of normalizer.
    #
    # This counts every day in the window from the bar rows, so it shows the
    # true roster size even for months the flag covers. The flag marks days,
    # not months, and the table is the reader's view of where it bites.
    print("\n=== roster health: live events per month")
    bars_uniq = bars[["event_id", "date"]].drop_duplicates()
    bars_uniq["month"] = bars_uniq["date"].dt.to_period("M")
    per_month = bars_uniq.groupby("month")["event_id"].nunique()
    live_all = per.groupby("date")["event_id"].nunique().reindex(
        days, fill_value=0)
    ml = live_all.resample("MS").mean()
    for m in pd.period_range(args.start, args.end, freq="M"):
        nlv = ml[ml.index.to_period("M") == m]
        nlv = nlv.iloc[0] if len(nlv) else 0.0
        # 50 live events a day is where the roster stops being an average and
        # starts being a handful of markets; it flags, it does not filter.
        flag = "  STARVED" if nlv < 50 else ""
        print(f"  {m}  live/day={nlv:>6.0f}  "
              f"distinct={per_month.get(m, 0):>4}{flag}")

    x = idx.join(ci).dropna(subset=["IDX_sum", "GPRD"])
    print(f"\n{len(x):,} days, "
          f"{x.index.min().date()} to {x.index.max().date()}")
    print(f"  peak correlation at lag +2: "
          f"{xcorr(x['IDX_sum'], x['GPRD'])[2]:+.3f}")

    print(f"\n=== contemporaneous correlation with GPRD (n={len(x):,})")
    print(f"{'index':<12} {'daily':>8} {'weekly':>8} {'log-daily':>10}"
          f" {'spearman':>10}")
    for col in ("IDX_sum", "R_sum", "IDX_threat", "IDX_act",
                "IDX_signed", "n_moving", "n_live"):
        if col not in x:
            continue
        wk = x[col].resample("W").sum().corr(x["GPRD"].resample("W").mean())
        # CI carries a handful of zero-article days (holidays). log(0) is
        # -inf and poisons the whole correlation, so drop them here.
        lg = np.log1p(x[col].clip(lower=0)).corr(
            np.log(x["GPRD"].where(x["GPRD"] > 0)))
        print(f"{col:<12} {x[col].corr(x['GPRD']):>8.3f} {wk:>8.3f} "
              f"{lg:>10.3f} {x[col].corr(x['GPRD'], method='spearman'):>10.3f}")

    print("\n=== the 2x2: threats should track GPRD_THREAT, acts GPRD_ACT")
    print("  by judge category -- what the event is about")
    print(f"  {'':<14} {'GPRD_THREAT':>12} {'GPRD_ACT':>10}   diagonal")
    two_by_two(x, ("IDX_threat", "IDX_act"))
    print("\n  by mechanism -- what happened to the market")
    print(f"  {'':<14} {'GPRD_THREAT':>12} {'GPRD_ACT':>10}   diagonal")
    two_by_two(x, ("IDX_open", "IDX_resolve"))

    periods(x)

    da, dg = deseason(x["IDX_sum"], x["GPRD"])
    dcc = xcorr(da, dg)
    print(f"\n=== weekly cycle: GPRD is a 5-day series, the market is 7-day")
    print(f"  peak with the day-of-week cycle left in    "
          f"{xcorr(x['IDX_sum'], x['GPRD']).max():+.3f} "
          f"at lag {xcorr(x['IDX_sum'], x['GPRD']).idxmax():+d}")
    print(f"  peak with it divided out of both series    "
          f"{dcc.max():+.3f} at lag {dcc.idxmax():+d}")
    print("  most of that gap is in-sample fit: with factors from training "
          "only,\n  the holdout gain is +0.004 at the peak and +0.047 one lag "
          "earlier.")

    print("\n=== cross-correlogram, IDX_sum vs GPRD "
          "(positive lag = market leads news)")
    cc = xcorr(x["IDX_sum"], x["GPRD"])
    for lag in range(-5, 6):
        bar = "#" * max(0, int(abs(cc[lag]) * 60))
        print(f"  {lag:>+3d}  {cc[lag]:>+6.3f}  {bar}")

    # The two mechanical halves carry the lead differently, and the shapes
    # are the argument. Open-market revision is anticipation: it correlates
    # broadly, including at negative lags where the news moves the market.
    # Resolution is realization: the market settles on the day the thing
    # happens, and the papers write it up afterwards, so it should sit flat
    # and then spike at one lag. A broad hump on that side would suggest
    # persistence rather than a publication lag.
    #
    # The capped excursion broadens the open-market channel slightly, because
    # crediting part of the intraday path picks up movement the close hides.
    # No smoothing is applied, so the peak lag remains interpretable: the
    # reported uncapped diagnostic peaked at +1. Reporting both halves keeps
    # any residual shift visible instead of over-interpreting the peak.
    if {"IDX_open", "IDX_resolve"} <= set(x.columns):
        co, cr = xcorr(x["IDX_open"], x["GPRD"]), xcorr(x["IDX_resolve"],
                                                        x["GPRD"])
        share = 100 * x["IDX_resolve"].sum() / x["IDX_sum"].sum()
        print(f"\n=== anticipation vs realization "
              f"(resolution is {share:.1f}% of the mass)")
        print(f"  {'lag':>4}{'open':>9}{'resolve':>10}")
        for lag in range(-3, 7):
            print(f"  {lag:>+4d}{co[lag]:>9.3f}{cr[lag]:>10.3f}   "
                  f"{'#' * max(0, int(co[lag] * 50)):<18}"
                  f"{'+' * max(0, int(cr[lag] * 50))}")
    print(f"  peak at lag {cc.idxmax():+d} (r={cc.max():+.3f})")

    if args.reliability:
        reliability(per, judg, days, x, n_iter=args.iters)

    if args.holdout:
        holdout(x, args.split)
        episodes(x)

    if args.regress:
        print("\n=== distributed lag: GPRD on 10 lags of IDX_sum "
              "(Newey-West)")
        distributed_lag(x["IDX_sum"], x["GPRD"])
        print("\n=== country-month panel against CI's GPRC series")
        country_panel(per, judg)


if __name__ == "__main__":
    main()
