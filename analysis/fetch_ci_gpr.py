"""Download the Caldara-Iacoviello GPR index and write a clean daily CSV.

This is the target series. The project compares its own market-derived index
against this one. Nothing in the index design is testable until this file
exists on disk.

Two files come from the author's site.

  data_gpr_daily_recent.xls  the daily index, 1985-01-01 forward
  data_gpr_export.xls        the monthly index, plus the eight category
                             shares (SHAREH_CAT_1..8) and the country series

Output: data/CI/gpr_daily.csv with one row for each calendar day.

Usage:
    python analysis/fetch_ci_gpr.py
"""

import os

import pandas as pd
import requests

BASE = "https://www.matteoiacoviello.com/gpr_files"
FILES = ("data_gpr_daily_recent.xls", "data_gpr_export.xls")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/CI")

# GPRD is the headline index. ACT and THREAT are the two halves that Caldara
# and Iacoviello report separately, and the split is the closest thing the
# newspaper index has to our own realized-versus-anticipated distinction.
KEEP = ["date", "GPRD", "GPRD_ACT", "GPRD_THREAT", "GPRD_MA7", "GPRD_MA30",
        "N10D"]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in FILES:
        path = os.path.join(OUT_DIR, name)
        resp = requests.get(f"{BASE}/{name}", timeout=60)
        resp.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(resp.content)
        print(f"  {name}: {len(resp.content):,} bytes")

    daily = pd.read_excel(os.path.join(OUT_DIR, FILES[0]))
    daily = daily[KEEP].dropna(subset=["date"])
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    out = os.path.join(OUT_DIR, "gpr_daily.csv")
    daily.to_csv(out, index=False)
    print(f"wrote {out}: {len(daily):,} days, "
          f"{daily['date'].min()} to {daily['date'].max()}")


if __name__ == "__main__":
    main()
