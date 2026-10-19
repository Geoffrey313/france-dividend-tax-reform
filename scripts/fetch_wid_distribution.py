#!/usr/bin/env python3
"""Refresh the committed WID distributional input from the official country bulk files."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.common import paths
from src.data import wid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory containing WID_data_<CC>.csv; omit to download from WID.world",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.DATA / "wid_distribution_2010_2022.csv",
    )
    args = parser.parse_args()
    if args.source_dir:
        frame = wid.build_from_directory(args.source_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    else:
        frame = wid.download(args.output)
    print(f"wrote {args.output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
