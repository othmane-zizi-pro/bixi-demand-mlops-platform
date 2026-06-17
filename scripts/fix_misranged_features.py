"""One-off cleanup: rewrite the Phase-1 feature tables to their intended date range.

Phase-1 exported some feature files with date-range spillover (the arrival files
span extra months; the 2024 files spill a few hours/days into 2025). The *data*
is correct once filtered, so this utility downloads each file, filters it to the
intended (year, months) for its split, and re-uploads a clean canonical copy.

By default it writes to a clean prefix (``processed-data-clean/``) so the original
Phase-1 outputs are untouched; pass ``--overwrite`` to replace in place.

The training pipeline already filters at load time (see ``bixi.data.load_split``),
so this is for a clean source-of-truth / data dictionary, not a correctness fix.

Usage:
  python scripts/fix_misranged_features.py            # dry run (report only)
  python scripts/fix_misranged_features.py --apply    # write clean copies
  python scripts/fix_misranged_features.py --apply --overwrite
"""

from __future__ import annotations

import argparse
import io as _io
import sys

import pandas as pd

sys.path.insert(0, "src")
from bixi import config, io  # noqa: E402


def run(apply: bool, overwrite: bool) -> None:
    clean_prefix = config.DATA_PREFIX if overwrite else "processed-data-clean"
    for target in config.TARGETS:
        for split, spec in config.split_specs(target).items():
            src_key = f"{config.DATA_PREFIX}/{spec.file_stem}.parquet"
            df = io.read_parquet_s3(src_key, bucket=config.DATA_BUCKET)
            before = len(df)
            ts = pd.to_datetime(df[config.TIME_COL])
            mask = ts.dt.year == spec.year
            if spec.months is not None:
                mask &= ts.dt.month.isin(spec.months)
            clean = df.loc[mask].reset_index(drop=True)
            print(f"{spec.file_stem}: {before:,} -> {len(clean):,} rows "
                  f"(year={spec.year}, months={spec.months})")
            if apply:
                buf = _io.BytesIO(); clean.to_parquet(buf, index=False)
                dst = f"{clean_prefix}/{spec.file_stem}.parquet"
                io.put_bytes(dst, buf.getvalue(), bucket=config.DATA_BUCKET)
                print(f"   wrote s3://{config.DATA_BUCKET}/{dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write cleaned files")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite originals in processed-data/ (default: processed-data-clean/)")
    args = ap.parse_args()
    run(apply=args.apply, overwrite=args.overwrite)
