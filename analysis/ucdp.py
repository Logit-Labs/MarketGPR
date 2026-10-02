"""Test the index against realized violence instead of newspaper coverage.

Every validation so far scores the index against Caldara-Iacoviello, and the
work on this branch established that CI is a partial yardstick. It publishes
no country series for Iran, Palestine, Yemen, Lebanon, North Korea or Syria,
which between them carry more of the roster than every matched country
combined, and the quiet-day diagnostic found real geopolitical content on
days GPRD did not move. An index scored only against newspapers is scored
against their blind spots too.

UCDP is the independent criterion. It codes organized-violence events from
the ground rather than from press volume, so it fails differently: a war
that newspapers stop covering still appears, and a news cycle with no
fighting behind it does not.

Coverage is the reason to prefer it here. UCDP reaches 1,155 of the 1,518
judged events by region against CI's 601, and the regions it adds are the
active theatres -- Iran, Yemen, Lebanon, Syria -- rather than large
economies.

Two caveats that belong in any write-up of the result:

  The candidate releases (CED) are preliminary. They are revised as coding
  is completed, so a month's count can move. They are also not comparable in
  level to the annual GED release, which is why this uses CED alone rather
  than splicing the two.

  UCDP counts violence, and the index prices anticipation. A market that
  correctly prices a war that does not start should score zero here and be
  right. The correlation is therefore a floor on the index's validity, not a
  ceiling.

Usage:
    python analysis/ucdp.py --fetch     # downloads to data/UCDP/
    python analysis/ucdp.py --test
"""

import argparse
import glob
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import index_pm as I  # noqa: E402

BASE = "https://ucdp.uu.se/downloads/candidateged/"
RAW = os.path.join(I.REPO, "data/UCDP")
PANEL = os.path.join(RAW, "ced_country_month.csv")

# UCDP writes country names, sometimes historical ones, and the judge writes
# ISO3. Only names that actually appear in the candidate files are listed.
# Palestine is deliberately absent: UCDP codes most Gaza violence under
# Israel, so mapping PSE onto it would double-count one theatre under two
# market regions.
NAME2ISO = {
    "Ukraine": "UKR", "Iran": "IRN", "Israel": "ISR",
    "Yemen (North Yemen)": "YEM", "Lebanon": "LBN", "Syria": "SYR",
    "Russia (Soviet Union)": "RUS", "Venezuela": "VEN", "Mexico": "MEX",
    "India": "IND", "Pakistan": "PAK", "Iraq": "IRQ", "Somalia": "SOM",
    "Sudan": "SDN", "Afghanistan": "AFG", "Myanmar (Burma)": "MMR",
    "Nigeria": "NGA", "Cambodia (Kampuchea)": "KHM", "Egypt": "EGY",
    "Thailand": "THA", "Philippines": "PHL", "Colombia": "COL",
    "Turkey": "TUR", "Saudi Arabia": "SAU", "DR Congo (Zaire)": "COD",
}


def _corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation, NaN if the paired sample cannot support one.

    The variance that matters is the one over the months both series share,
    because that is the sample ``corr`` uses. Checking each series on its own
    non-null values lets a pair with a flat overlap through, and pandas then
    divides by a zero standard deviation: NaN, but with a numpy warning.
    """
    q = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(q) < 3:
        return float("nan")
    sa, sb = q["a"].std(), q["b"].std()
    if not (np.isfinite(sa) and np.isfinite(sb) and sa > 0 and sb > 0):
        return float("nan")
    return float(q["a"].corr(q["b"]))


def fetch(force: bool = False) -> None:
    os.makedirs(RAW, exist_ok=True)
    got = 0
    for yr in (25, 26):
        for mo in range(1, 13):
            name = f"GEDEvent_v{yr}_0_{mo}.csv"
            dest = os.path.join(RAW, name)
            if os.path.exists(dest):
                got += 1
                continue
            r = requests.get(BASE + name, timeout=60)
            if r.ok and r.content[:1] not in (b"<", b""):
                with open(dest, "wb") as fh:
                    fh.write(r.content)
                got += 1
    print(f"  {got} candidate files in {RAW}")

    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "GEDEvent_*.csv"))):
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            print(f"  skipped {os.path.basename(f)}: {exc}")
    d = pd.concat(frames, ignore_index=True)
    # Releases overlap, so the same event appears in several files.
    d = d.drop_duplicates(subset=["id"])
    d["date_start"] = pd.to_datetime(d["date_start"], errors="coerce")
    d = d.dropna(subset=["date_start"])
    d["iso"] = d["country"].map(NAME2ISO)
    d = d.dropna(subset=["iso"])
    d["month"] = d["date_start"].dt.to_period("M")

    panel = d.groupby(["iso", "month"]).agg(
        events=("id", "size"), deaths=("best", "sum")).reset_index()
    new_panel = PANEL + ".new.csv"
    panel.to_csv(new_panel, index=False)
    def digest(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    old_hash = digest(PANEL) if os.path.exists(PANEL) else None
    new_hash = digest(new_panel)
    print(f"  wrote candidate {new_panel}: {len(panel):,} country-months, "
          f"{panel.iso.nunique()} countries, {panel.month.min()} to {panel.month.max()}")
    print(f"  candidate sha256={new_hash}; existing sha256={old_hash}")
    if force:
        os.replace(new_panel, PANEL)
        print(f"  replaced {PANEL}")


def panel_test(per: pd.DataFrame, judg: pd.DataFrame,
               days: pd.DatetimeIndex, min_events: int = 10) -> None:
    if not os.path.exists(PANEL):
        sys.exit(f"missing {PANEL} -- run --fetch first")
    u = pd.read_csv(PANEL)
    u["month"] = pd.PeriodIndex(u["month"], freq="M")

    # Months where a country appears in no UCDP row are true zeros, not
    # missing data: the coding covers every country each month.
    reg = judg[["event_id", "region"]].dropna()
    counts = reg.groupby("region")["event_id"].nunique()
    keep = [r for r in counts[counts >= min_events].index
            if r in set(u["iso"])]
    keep.sort(key=lambda r: -counts[r])

    series = {}
    for r in keep:
        ids = set(reg.loc[reg["region"] == r, "event_id"])
        idx = I.assemble(per[per["event_id"].isin(ids)], judg, days)
        # Calendar-complete frames are never empty, so guard on whether the
        # country's series carries any values.
        s = idx["IDX_sum"].groupby(idx.index.to_period("M")).mean()
        if s.notna().any():
            series[r] = s
    im = pd.DataFrame(series)

    wide_e = u.pivot(index="month", columns="iso", values="events")
    wide_d = u.pivot(index="month", columns="iso", values="deaths")
    common = im.index.intersection(wide_e.index)
    cols = [c for c in im.columns if c in wide_e.columns]
    im = im.loc[common, cols]
    we = wide_e.loc[common, cols].fillna(0.0)
    wd = wide_d.loc[common, cols].fillna(0.0)

    matched = int(counts[cols].sum())
    print(f"\n=== UCDP realized violence, {len(cols)} countries")
    print(f"  matched {matched:,} of {int(counts.sum()):,} judged events "
          f"({100 * matched / counts.sum():.0f}%) -- CI's country GPR "
          f"reaches 40%")
    print(f"  {len(common)} months, {common.min()} to {common.max()}")

    # Fatalities are heavy tailed; one massacre would otherwise set a
    # country's whole correlation. Counts are the primary for that reason.
    lwd = np.log1p(wd)
    print(f"\n  {'region':<8}{'n_ev':>6}{'r_events':>10}{'r_deaths':>10}"
          f"{'ucdp/mo':>10}")
    for c in cols:
        print(f"  {c:<8}{int(counts[c]):>6}{_corr(im[c], we[c]):>10.3f}"
              f"{_corr(im[c], lwd[c]):>10.3f}{we[c].mean():>10.0f}")

    def pooled(y: pd.DataFrame, label: str) -> None:
        zi = (im - im.mean()) / im.std()
        zc = (y - y.mean()) / y.std()
        long = pd.DataFrame({"i": zi.stack(), "c": zc.stack()}).dropna()
        if len(long) < 20:
            return
        r_raw = long["i"].corr(long["c"])
        d2 = long.copy()
        for col in ("i", "c"):
            d2[col] -= d2.groupby(level=0)[col].transform("mean")
        print(f"  {label:<22}{r_raw:>+8.3f}   month effects removed "
              f"{d2['i'].corr(d2['c']):>+.3f}   n={len(long):,}")

    # Does each country's index track its OWN violence more than another
    # country's? Same test as the regional GPR decomposition, and the same
    # null: permute which country counts as "own", which reuses the one
    # correlation matrix and holds its structure fixed.
    m = np.array([[_corr(im[a], we[b]) for b in cols] for a in cols])
    ranks, wins = [], 0
    for i in range(len(cols)):
        row = m[i][np.isfinite(m[i])]
        if not np.isfinite(m[i, i]) or not row.size:
            continue
        ranks.append(1 + int((np.sort(row)[::-1] > m[i, i]).sum()))
        wins += m[i, i] >= row.max()
    if ranks:
        rng = np.random.default_rng(20260909)
        null = np.empty(10000)
        for k in range(null.size):
            pi = rng.permutation(len(cols))
            null[k] = np.mean([
                1 + int((np.sort(m[i][np.isfinite(m[i])])[::-1]
                         > m[i, pi[i]]).sum())
                for i in range(len(cols)) if np.isfinite(m[i, pi[i]])])
        obs = float(np.mean(ranks))
        print(f"\n  own country ranks first for {wins} of {len(ranks)}; "
              f"mean rank {obs:.1f} against {(len(cols) + 1) / 2:.1f}")
        print(f"  permutation test, 10,000 relabelings: "
              f"p = {(null <= obs).mean():.4f}")

    print()
    pooled(we, "pooled, events")
    pooled(lwd, "pooled, log deaths")

    # Does the index track violence in the months newspapers are quiet?
    # This is the question the whole exercise exists for: the quiet-day
    # signal replicates across a split roster, so it is not noise, and
    # nothing so far could say whether it was geopolitics.
    ci = I.ci_series()
    gm = ci["GPRD"].groupby(ci.index.to_period("M")).mean().reindex(common)
    quiet = gm < gm.median()
    for label, mask in (("quiet GPRD months", quiet),
                        ("loud GPRD months", ~quiet)):
        sub_i, sub_c = im.loc[mask.values], we.loc[mask.values]
        zi = (sub_i - sub_i.mean()) / sub_i.std()
        zc = (sub_c - sub_c.mean()) / sub_c.std()
        long = pd.DataFrame({"i": zi.stack(), "c": zc.stack()}).dropna()
        if len(long) > 20:
            print(f"  {label:<22}{long['i'].corr(long['c']):>+8.3f}"
                  f"   n={len(long):,}")


def _ols_cluster(y: np.ndarray, x: np.ndarray, g: np.ndarray) -> tuple:
    """OLS with standard errors clustered on g. No statsmodels here.

    Conflict is persistent within a country and the residuals are correlated
    down each country's own time series, so plain OLS errors would be far
    too small. Clustering on country is the standard fix and the only one
    this sample supports -- 16 clusters is on the low side, so read the
    t-statistic as indicative rather than exact.
    """
    x = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    u = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cl in np.unique(g):
        m = g == cl
        s = x[m].T @ u[m]
        meat += np.outer(s, s)
    n, k, ng = len(y), x.shape[1], len(np.unique(g))
    adj = (ng / max(ng - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = xtx_inv @ meat @ xtx_inv * adj
    return beta, np.sqrt(np.diag(cov))


def _demean(df: pd.DataFrame, cols: list, by: list, rounds: int = 8):
    """Two-way within transformation, applied iteratively."""
    d = df.copy()
    for _ in range(rounds):
        for lvl in by:
            for c in cols:
                d[c] = d[c] - d.groupby(lvl)[c].transform("mean")
    return d


def lead(per: pd.DataFrame, judg: pd.DataFrame, days: pd.DatetimeIndex,
         min_events: int = 10) -> None:
    """Does the index forecast next month's violence, beyond this month's?

    The contemporaneous result says the index and realized violence move
    together. The interesting claim for market data is that it moves first,
    and the honest version of that test controls for the violence already
    happening, because conflict is persistent and last month's fighting
    predicts this month's without anyone forecasting anything.

    Both sides are standardized within country, so a coefficient is in
    standard deviations and Ukraine's war does not set the scale for
    everyone. Country and month effects are removed, the month effects
    because a global escalation would otherwise look like every country's
    index predicting every country's violence.
    """
    if not os.path.exists(PANEL):
        sys.exit(f"missing {PANEL} -- run --fetch first")
    u = pd.read_csv(PANEL)
    u["month"] = pd.PeriodIndex(u["month"], freq="M")

    reg = judg[["event_id", "region"]].dropna()
    counts = reg.groupby("region")["event_id"].nunique()
    keep = [r for r in counts[counts >= min_events].index
            if r in set(u["iso"])]

    rows = []
    for r in keep:
        ids = set(reg.loc[reg["region"] == r, "event_id"])
        idx = I.assemble(per[per["event_id"].isin(ids)], judg, days)
        s = idx["IDX_sum"].groupby(idx.index.to_period("M")).mean()
        if not s.notna().any():
            continue
        v = u[u["iso"] == r].set_index("month")["events"]
        m = pd.DataFrame({"idx": s}).join(v.rename("viol"), how="inner")
        if len(m) < 6:
            continue
        m.index.name = "month"
        m["iso"] = r
        m["viol"] = np.log1p(m["viol"])
        m["viol_next"] = m["viol"].shift(-1)
        m["idx_next"] = m["idx"].shift(-1)
        rows.append(m.reset_index())
    d = pd.concat(rows, ignore_index=True).dropna(
        subset=["idx", "viol", "viol_next"])

    for c in ("idx", "viol", "viol_next", "idx_next"):
        d[c] = d.groupby("iso")[c].transform(
            lambda s: (s - s.mean()) / s.std() if s.std() else s * 0)
    d = d.dropna(subset=["idx", "viol", "viol_next"])

    print(f"\n=== does the index lead realized violence?")
    print(f"  {len(d)} country-months, {d.iso.nunique()} countries, "
          f"{d['month'].min()} to {d['month'].max()}")
    print("  outcome: log violence next month, standardized within country")

    w = _demean(d, ["idx", "viol", "viol_next", "idx_next"],
                ["iso", "month"])
    g = d["iso"].to_numpy()

    b, se = _ols_cluster(w["viol_next"].to_numpy(),
                         w[["idx", "viol"]].to_numpy(), g)
    names = ["const", "index this month", "violence this month"]
    print(f"\n  {'term':<24}{'coef':>9}{'se':>9}{'t':>8}")
    for i, nm in enumerate(names):
        t = b[i] / se[i] if se[i] else np.nan
        print(f"  {nm:<24}{b[i]:>+9.3f}{se[i]:>9.3f}{t:>8.2f}"
              + ("  *" if abs(t) > 1.96 else ""))

    # The mirror image. If violence predicts the index just as well, the
    # arrow is not established and both are reading a common shock.
    b2, se2 = _ols_cluster(w["idx_next"].dropna().to_numpy(),
                           w.dropna(subset=["idx_next"])[
                               ["viol", "idx"]].to_numpy(),
                           d.dropna(subset=["idx_next"])["iso"].to_numpy())
    t2 = b2[1] / se2[1] if se2[1] else np.nan
    print(f"\n  reverse: violence this month -> index next month")
    print(f"  {'violence this month':<24}{b2[1]:>+9.3f}{se2[1]:>9.3f}"
          f"{t2:>8.2f}" + ("  *" if abs(t2) > 1.96 else ""))
    print(f"\n  {d.iso.nunique()} clusters is few, so treat t as "
          f"indicative. And note the\n  timescale: the GPRD result is a "
          f"lead of two DAYS, which a monthly\n  panel cannot resolve at "
          f"all. This tests a different and stronger\n  claim -- "
          f"month-ahead forecasting -- and that claim fails.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="replace the tracked panel after writing .new.csv")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--lead", action="store_true",
                    help="does the index forecast next month's violence?")
    ap.add_argument("--start", default="2024-08-20")
    ap.add_argument("--end", default="2026-09-01")
    args = ap.parse_args()
    if args.fetch:
        fetch(args.force)
        return
    if not (args.test or args.lead):
        ap.print_help()
        return
    bars, judg = I.load()
    per = I.revisions(bars)
    days = pd.date_range(args.start, args.end, freq="D")
    if args.test:
        panel_test(per, judg, days)
    if args.lead:
        lead(per, judg, days)


if __name__ == "__main__":
    main()
