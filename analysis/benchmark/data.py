"""Index build, comparator acquisition, and the aligned analysis panel.

Three kinds of series are combined, and the difference matters for every
statistic computed downstream.

  calendar  A value on every calendar day. The index itself and CI's GPRD
            are these: both are flows of counts/revisions that exist on
            weekends.

  trading   A value only on days the venue was open: VIX, oil, the dollar,
            credit spreads. Carry-forward is the only defensible way to put
            these on a calendar grid, and a level correlation across a
            forward-filled series is legitimate because the level really did
            persist over the weekend. A *change* correlation is not: the
            forward-filled days carry zeros that were never observed, which
            attenuates the correlation mechanically. So changes are computed
            on the native observations only and left missing on the filled
            days, and the tests that use changes inherit the trading mask.

  monthly   CI's monthly GPR family, country indices, policy uncertainty,
            and the realized-violence panel.

Everything is cached under `data/benchmark/`, and `manifest.json` records
the source URL and the retrieval date for each file. CI and the FRED series
are revised, so a table of numbers without that date is not reproducible.

The index is built through `index_pm` rather than reimplemented. Its
published defaults are the object under test, so the benchmark must not be
able to disagree with the pipeline by accident.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "analysis"))

import index_pm as I  # noqa: E402

CACHE = os.path.join(REPO, "data", "benchmark")
RAW = os.path.join(CACHE, "raw")
OUT_ROOT = os.path.join(os.path.expanduser("~"), "Downloads",
                        "Test Outputs", "Final Benchmark")

# assemble() needs TRAIL=90 days of roster before the trailing normalizer is
# defined; min_periods=20 means the first published value is 2024-08-20. The
# build therefore starts earlier than the sample it publishes.
BUILD_START = "2024-08-01"
START = "2024-08-20"
END = "2026-09-01"
LAG = 2                 # the lead the project reports
UA = {"User-Agent": "Mozilla/5.0 (marketgpr benchmark)"}

# name -> (FRED series id, description). Every series is daily except the two
# noted; FRED marks a missing observation with ".".
FRED = {
    "VIX": ("VIXCLS", "CBOE S&P 500 implied volatility, close"),
    "VXN": ("VXNCLS", "CBOE Nasdaq-100 implied volatility, close"),
    "OVX": ("OVXCLS", "CBOE crude oil ETF implied volatility, close"),
    "GVZ": ("GVZCLS", "CBOE gold ETF implied volatility, close"),
    "EPU_US": ("USEPUINDXD", "US economic policy uncertainty, daily news-based"),
    "WTI": ("DCOILWTICO", "Cushing WTI crude, spot, USD/bbl"),
    "DXY_BROAD": ("DTWEXBGS", "Trade-weighted US dollar, broad goods and services"),
    "DXY_EME": ("DTWEXEMEGS", "Trade-weighted US dollar, emerging markets"),
    "DGS10": ("DGS10", "10-year Treasury constant maturity, percent"),
    "DGS2": ("DGS2", "2-year Treasury constant maturity, percent"),
    "HY_OAS": ("BAMLH0A0HYM2", "ICE BofA US high yield option-adjusted spread"),
    "T10YIE": ("T10YIE", "10-year breakeven inflation rate"),
    "SP500": ("SP500", "S&P 500 index level, close"),
    "JPY": ("DEXJPUS", "Japanese yen per USD"),
    "CNY": ("DEXCHUS", "Chinese yuan per USD"),
    "NFCI": ("NFCI", "Chicago Fed national financial conditions index, weekly"),
    "STLFSI": ("STLFSI4", "St. Louis Fed financial stress index, weekly"),
}

# Which FRED series are levels the rest of the world reads as a level
# (VIX, spreads, rates, prices) and which are prices whose change is the
# informative object. Only used to label tables; every series gets both a
# level and a change column in the panel.
LEVEL_KEYS = {"VIX", "VXN", "OVX", "GVZ", "EPU_US", "HY_OAS", "DGS10",
              "DGS2", "T10YIE", "NFCI", "STLFSI", "DXY_BROAD", "DXY_EME",
              "SP500", "WTI", "BRENT", "JPY", "CNY"}

# The quarterly WUI file the project's own documentation links to stops at
# 2023Q4, which shares no day with this sample. The monthly file is the one
# that overlaps, so the monthly version is what gets used.
WUI_MONTHLY_URL = ("https://www.worlduncertaintyindex.com/wp-content/uploads/"
                   "2026/09/WUI_M_dataset_2026_08.xlsx")
EPU_MONTHLY_URL = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
GEPU_MONTHLY_URL = "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.csv"
GEPU_FRED = ("GEPUCURRENT", "Global economic policy uncertainty, monthly")
# FRED series that are monthly, not daily. Brent is here rather than in FRED
# above because the vintage served by fredgraph is monthly: as a daily column
# it arrives 97% missing, and a benchmark that correlates on 24 observations
# without saying so is worse than one that leaves the series out of the daily
# panel and carries it at its own frequency.
MONTHLY_FRED = {
    "BRENT_M": ("MCOILBRENTEU", "Brent crude, spot, USD/bbl, monthly"),
    "EPU_GLOBAL_FRED": ("GEPUCURRENT", GEPU_FRED[1]),
}

# CI's monthly category columns are *shares* of the index, not levels: CI
# publishes the eight topic weights, and SHAREH_CAT_1..8 is where the
# composition comparison reads from.
CI_CATS = {
    "SHAREH_CAT_1": "war",
    "SHAREH_CAT_2": "diplomacy_about_conflict",
    "SHAREH_CAT_3": "terror",
    "SHAREH_CAT_4": "nuclear",
    "SHAREH_CAT_5": "missile_test",
    "SHAREH_CAT_6": "domestic_insecurity",
    "SHAREH_CAT_7": "civil_war",
    "SHAREH_CAT_8": "military_buildup",
}

MANIFEST: dict = {}


# ---------------------------------------------------------------------------
# retrieval


def _get(url: str, tries: int = 4, pause: float = 3.0) -> bytes | None:
    """Fetch a URL, returning None after the last failed attempt.

    The benchmark must not die because one comparator's host is down. A
    series that cannot be retrieved is recorded as unavailable in the
    manifest and left out of the panel, which keeps every released table
    reproducible from the sources that did answer.
    """
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as exc:            # noqa: BLE001 - report, do not die
            last = exc
            if k < tries - 1:
                time.sleep(pause * (k + 1))
    MANIFEST.setdefault("unavailable", {})[url] = f"{type(last).__name__}: {last}"
    return None


def _cached(url: str, name: str) -> str | None:
    """Download *url* once and register the cache on every run."""
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, name)
    if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
        blob = _get(url)
        if blob is None:
            return None
        with open(dest, "wb") as fh:
            fh.write(blob)
    h = hashlib.sha256()
    with open(dest, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    MANIFEST.setdefault("sources", {})[name] = {
        "name": name, "path": dest, "bytes": os.path.getsize(dest),
        "sha256": h.hexdigest(), "url": url,
        "mtime": os.path.getmtime(dest),
    }
    return dest


def _fred_csv(series: str) -> pd.Series:
    """One FRED series as a float series indexed by date.

    Parsed by position rather than by column name: FRED has shipped both
    DATE/VALUE and observation_date/<series> headers, and the shape is
    always the same two columns.
    """
    path = _cached(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                   f"fred_{series}.csv")
    if path is None:
        return pd.Series(dtype=float)
    d = pd.read_csv(path)
    d = d.rename(columns={d.columns[0]: "date", d.columns[1]: "value"})
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    return d.dropna().set_index("date")["value"].astype(float)


def _fred_all() -> pd.DataFrame:
    out = {}
    for name, (sid, _desc) in FRED.items():
        s = _fred_csv(sid)
        if len(s):
            out[name] = s
    return pd.DataFrame(out).sort_index()


# ---------------------------------------------------------------------------
# the index itself


def build_index() -> pd.DataFrame:
    """The published index over the full build window, one row per day.

    Returned from 2024-08-01 so the caller can see the warm-up, with the
    sample boundary marked by the first non-null IDX_sum.
    """
    bars, judg = I.load()
    per = I.revisions(bars)
    days = pd.date_range(BUILD_START, END, freq="D")
    idx = I.assemble(per, judg, days)
    idx.index.name = "date"
    return idx


def _index_cached() -> pd.DataFrame:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "index_daily.csv")
    idx = build_index()
    idx.to_csv(path, date_format="%Y-%m-%d")
    return idx


# ---------------------------------------------------------------------------
# monthly and quarterly comparators


def ci_monthly() -> pd.DataFrame:
    """CI's monthly export: aggregate, threat/act split, and topic shares.

    GPR is the index on 1985:2019=100; GPRT and GPRA are the threat and act
    halves; GPRH is the long historical backcast. The SHAREH_CAT_* columns
    are the eight topic weights that make up the level.
    """
    d = pd.read_excel(os.path.join(REPO, "data/CI/data_gpr_export.xls"))
    keep = ["GPR", "GPRT", "GPRA", "GPRH", "N10"] + list(CI_CATS)
    out = d[["month"] + [c for c in keep if c in d.columns]].copy()
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M")
    out = out.set_index("month")
    out = out.rename(columns=CI_CATS)
    return out.apply(pd.to_numeric, errors="coerce")


def ci_country() -> pd.DataFrame:
    """CI's monthly country-level GPR, indexed by month, columns ISO3."""
    d = pd.read_excel(os.path.join(REPO, "data/CI/data_gpr_export.xls"))
    cols = [c for c in d.columns if c.upper().startswith("GPRC_")]
    out = d[["month"] + cols].copy()
    out["month"] = pd.to_datetime(out["month"]).dt.to_period("M")
    out = out.set_index("month")
    out.columns = [c.replace("GPRC_", "") for c in cols]
    return out.apply(pd.to_numeric, errors="coerce")


def epu_monthly() -> pd.DataFrame:
    """Baker-Bloom-Davis policy uncertainty, monthly, US and global.

    EPU counts newspaper references to economic policy uncertainty, so it is
    the closest thing in the set to a second opinion on CI's method: same
    kind of source, same kind of count, different subject matter. That makes
    it the useful control for "is the index tracking geopolitical risk, or
    just tracking newspaper volume".
    """
    out = {}
    us = _cached(EPU_MONTHLY_URL, "epu_us_monthly.csv")
    if us is not None:
        d = pd.read_csv(us).rename(columns={
            "News_Based_Policy_Uncert_Index": "EPU_US_M"})
        d["month"] = pd.to_datetime(
            dict(year=d["Year"], month=d["Month"], day=1)).dt.to_period("M")
        out["EPU_US_M"] = d.set_index("month")["EPU_US_M"]
    ge = _cached(GEPU_MONTHLY_URL, "gepu_monthly.csv")
    if ge is not None:
        d = pd.read_csv(ge)
        d["month"] = pd.to_datetime(
            dict(year=d["Year"], month=d["Month"], day=1)).dt.to_period("M")
        out["EPU_GLOBAL_M"] = d.set_index("month")[d.columns[2]]
    return pd.DataFrame(out).sort_index()


def fred_monthly() -> pd.DataFrame:
    """FRED series published at monthly frequency, indexed by month."""
    out = {}
    for name, (sid, _desc) in MONTHLY_FRED.items():
        s = _fred_csv(sid)
        if len(s):
            out[name] = pd.Series(s.to_numpy(float),
                                  index=s.index.to_period("M"))
    return pd.DataFrame(out).sort_index()


def wui_monthly() -> pd.DataFrame:
    """The World Uncertainty Index: global aggregate and country series.

    Built from EIU country-report text, so it is a third independent
    text-based measure alongside CI's newspaper count and the policy
    uncertainty index. It is monthly, and is only ever used at monthly
    frequency: interpolating it to days would invent variation that the
    source does not have.
    """
    path = _cached(WUI_MONTHLY_URL, "wui_monthly.xlsx")
    if path is None:
        return pd.DataFrame()
    glob = pd.read_excel(path, sheet_name="F1", header=None)
    # The sheet carries two header rows above the data; the series name is
    # in row 1 and the observations start in row 2.
    g = glob.iloc[2:, :2].copy()
    g.columns = ["date", "WUI"]
    g["date"] = pd.to_datetime(
        g["date"].astype(str).str.slice(0, 10), errors="coerce",
        format="%Y-%m-%d")
    g = g.dropna(subset=["date"]).set_index("date")
    g.index = g.index.to_period("M")
    g["WUI"] = pd.to_numeric(g["WUI"], errors="coerce")

    ctry = pd.read_excel(path, sheet_name="T1")
    ctry["date"] = pd.to_datetime(ctry["date"])
    ctry = ctry.set_index(ctry.pop("date").dt.to_period("M"))
    ctry = ctry.apply(pd.to_numeric, errors="coerce")
    ctry.columns = [f"WUI_{c}" for c in ctry.columns]
    return g.join(ctry, how="outer").sort_index()


def ucdp_monthly() -> pd.DataFrame:
    """Realized violence: UCDP candidate GED, deaths and events by country.

    This is the only comparator in the set that is not built from text or
    prices. It measures what actually happened, which is why it is the check
    that does not inherit CI's country blind spots. It is a country-month
    panel, not a series: summing it to a global total is a different object,
    and `ucdp_global` does that explicitly rather than leaving it to luck.
    """
    path = os.path.join(REPO, "data/UCDP/ced_country_month.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    d = pd.read_csv(path)
    d["month"] = pd.PeriodIndex(d["month"], freq="M")
    return d.set_index("month").sort_index()


def ucdp_global() -> pd.DataFrame:
    """UCDP summed over countries: one row per month."""
    d = ucdp_monthly()
    if not len(d):
        return d
    g = d.groupby(level=0)[["events", "deaths"]].sum()
    g.columns = ["ucdp_events", "ucdp_deaths"]
    return g.sort_index()


# ---------------------------------------------------------------------------
# the aligned panel


@dataclass
class Panel:
    """Everything a test needs, aligned on one calendar."""

    days: pd.DatetimeIndex
    daily: pd.DataFrame        # levels, calendar-daily (trading series filled)
    dchg: pd.DataFrame         # changes on native observations only
    obs: pd.DataFrame          # True where the level is a native observation
    idx: pd.DataFrame          # the index frame, including roster columns
    per: pd.DataFrame          # event-day revisions, for reliability
    judg: pd.DataFrame         # the judged roster
    monthly: pd.DataFrame      # monthly levels and flows
    wui: pd.DataFrame          # World Uncertainty Index, monthly
    ucdp: pd.DataFrame
    country: pd.DataFrame
    index_cols: list           # the index's own IDX_* channels
    aux_cols: list             # roster size and unnormalized revision mass
    external_cols: list        # comparators, all of them

    @property
    def sample(self) -> str:
        return f"{self.days.min().date()} to {self.days.max().date()}"

    def change(self, col: str) -> pd.Series:
        """Daily change, defined only where the source actually observed."""
        return self.dchg[col]

    def flow(self, col: str) -> pd.Series:
        """A non-negative flow series."""
        return self.daily[col]


def _trading_pairs(native: pd.DataFrame, days: pd.DatetimeIndex) -> tuple:
    """Split native observations into levels and native-only changes.

    The change is taken on the *raw* observation series, before anything is
    put on the calendar. Taking it after reindexing would pair a trading day
    with the missing weekend cell above it and produce NaN, silently dropping
    every Friday-to-Monday move — which is exactly where weekend geopolitical
    news lands, so the loss would not be random. Differencing first gives one
    Friday-to-Monday close-to-close change, and reindexing then places it on
    the Monday and leaves the weekend cells missing, which is what the change
    sample should be.
    """
    lvl = native.reindex(days).ffill()
    # Per column, not frame-wide: the FRED frame is a union index, so a
    # weekend row exists for the daily series and is NaN for VIX. A frame
    # diff would look up that NaN and produce a missing change again.
    chg = pd.DataFrame({c: native[c].dropna().diff()
                        for c in native.columns}).reindex(days)
    obs = native.reindex(days).notna()
    return lvl, chg, obs


def _monthly_panel(idx: pd.DataFrame, ci_daily: pd.DataFrame,
                   com: pd.DataFrame) -> pd.DataFrame:
    """Monthly means for levels and flows, at the calendar's own period.

    Means rather than sums for the level series: GPRD is already an index
    scaled to 100, so summing it would make the monthly number a function of
    month length. The index is a per-day normalized flow, so its monthly
    mean keeps the same units. `flow_*` sums are also kept for the tests
    that want an activity total rather than a rate.
    """
    m = idx[["IDX_sum", "IDX_threat", "IDX_act", "n_live", "n_moving"]].copy()
    m["GPRD"] = ci_daily["GPRD"]
    m["GPRD_ACT"] = ci_daily["GPRD_ACT"]
    m["GPRD_THREAT"] = ci_daily["GPRD_THREAT"]
    if len(com):
        m = m.join(com, how="left")
    g = m.groupby(m.index.to_period("M"))
    out = g.mean()
    out["flow_IDX_sum"] = g["IDX_sum"].sum()
    out["flow_GPRD"] = g["GPRD"].sum()
    out["flow_n_moving"] = g["n_moving"].sum()
    return out


def public_sample(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The published window: days the trailing normalizer is defined."""
    return days[(days >= START) & (days <= END)]


@functools.lru_cache(maxsize=1)
def panel() -> Panel:
    """Build the cached panel. Memoized: every test shares one build."""
    ci_daily = I.ci_series()
    ci_daily = ci_daily.reindex(
        pd.date_range(BUILD_START, END, freq="D")).ffill()
    # N10D is the raw article count GPRD is built from, and it is the right
    # control for "is the index tracking geopolitics or just newspaper
    # volume". index_pm's ci_series() drops it, so it is read here.
    _raw = pd.read_csv(I.CI_CSV, parse_dates=["date"]).set_index("date")
    if "N10D" in _raw.columns:
        ci_daily["N10D"] = _raw["N10D"].reindex(
            pd.date_range(BUILD_START, END, freq="D")).ffill()
    idx = _index_cached()

    com = _fred_all()

    full = pd.date_range(BUILD_START, END, freq="D")
    levels, chgs, obs = {}, {}, {}

    # The index is a calendar flow: it exists on every day.
    for col in idx.columns:
        levels[col] = idx[col]
        chgs["d_" + col] = idx[col].diff()
        obs[col] = idx[col].notna()
    for col in ci_daily.columns:
        levels[col] = ci_daily[col]
        chgs["d_" + col] = ci_daily[col].diff()
        obs[col] = ci_daily[col].notna()

    # Market series: levels are carried forward, changes are not.
    if len(com):
        lvl, chg, ob = _trading_pairs(com, full)
        for col in com.columns:
            levels[col] = lvl[col]
            chgs["d_" + col] = chg[col]
            obs[col] = ob[col]

    daily = pd.DataFrame(levels).reindex(full)
    dchg = pd.DataFrame(chgs).reindex(full)
    obs = pd.DataFrame(obs).reindex(full)

    days = public_sample(full)
    daily, dchg, obs = daily.loc[days], dchg.loc[days], obs.loc[days]

    na = daily.isna().mean()
    index_cols = [c for c in idx.columns if c.startswith("IDX_")
                  and daily[c].notna().mean() > .9]
    aux_cols = [c for c in ("n_live", "n_moving", "R_sum", "S_sum")
                if c in daily.columns]
    external_cols = [c for c in daily.columns
                     if c not in idx.columns and na[c] < .5]

    wui = wui_monthly()
    mo = _monthly_panel(idx, ci_daily, com).join(
        epu_monthly(), how="left").join(fred_monthly(), how="left").join(
        ci_monthly(), how="left")
    mo = mo.join(ucdp_global(), how="left")
    if len(wui):
        mo = mo.join(wui[["WUI"]], how="left")
    mo = mo[~mo.index.duplicated(keep="first")]

    bars, judg = I.load()
    per = I.revisions(bars)

    return Panel(days=days, daily=daily, dchg=dchg, obs=obs, idx=idx,
                 per=per, judg=judg, monthly=mo, wui=wui,
                 ucdp=ucdp_monthly(), country=ci_country(),
                 index_cols=index_cols, aux_cols=aux_cols,
                 external_cols=external_cols)


def write_manifest(extra: dict | None = None) -> str:
    """Record where every number came from, and when it was read."""
    os.makedirs(CACHE, exist_ok=True)
    p = panel()
    doc = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample": p.sample,
        "days": int(len(p.days)),
        "build_start": BUILD_START,
        "index_config": {
            "transform": I.TRANSFORM, "measure": I.MEASURE,
            "wick_cap": I.WICK_CAP, "agg": I.AGG, "norm": I.NORM,
            "trail": I.TRAIL, "smooth": I.SMOOTH_HALFLIFE,
            "conf_weight": I.CONF_WEIGHT,
            "composites": not I.EXCLUDE_COMPOSITES,
        },
        "sources": MANIFEST,
        "fred": {k: v[0] for k, v in FRED.items()},
        "ci_daily_csv": os.path.join(REPO, "data/CI/gpr_daily.csv"),
        "ci_monthly_xls": os.path.join(REPO, "data/CI/data_gpr_export.xls"),
        "ucdp": os.path.join(REPO, "data/UCDP/ced_country_month.csv"),
    }
    if extra:
        doc.update(extra)
    path = os.path.join(CACHE, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    return path


if __name__ == "__main__":
    p = panel()
    print(f"{len(p.days)} days, {p.sample}")
    print(f"index columns: {p.index_cols}")
    print(f"aux columns: {p.aux_cols}")
    print(f"external columns: {p.external_cols}")
    print(f"monthly: {p.monthly.shape}, wui: {p.wui.shape}, "
          f"ucdp: {p.ucdp.shape}")
    print(f"manifest: {write_manifest()}")
