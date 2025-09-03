#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None


def read_csv(path: Path):
    # Use pandas if available for robust CSV parsing, else fallback to csv
    if pd is not None:
        return pd.read_csv(path)
    # Fallback minimal parser
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert to a simple pandas-like object if pandas not available
    # but we will require pandas to write Excel. If not present, raise.
    raise RuntimeError("pandas is required to write Excel. Please install pandas.")


def main():
    parser = argparse.ArgumentParser(description="Merge per-site CSVs into a single Excel workbook")
    parser.add_argument(
        "--csv_dir",
        type=Path,
        default=Path("/home/zeyuli/Code/VisualAgentBench/VAB-WebArena-Lite/utils/lite_templates_csv"),
        help="Directory containing site_*.csv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/zeyuli/Code/VisualAgentBench/VAB-WebArena-Lite/utils/lite_doable.xlsx"),
        help="Output Excel file path",
    )
    args = parser.parse_args()

    site_files = {
        "gitlab": args.csv_dir / "site_gitlab.csv",
        "map": args.csv_dir / "site_map.csv",
        "reddit": args.csv_dir / "site_reddit.csv",
        "shopping_admin": args.csv_dir / "site_shopping_admin.csv",
        "shopping": args.csv_dir / "site_shopping.csv",
        "wikipedia": args.csv_dir / "site_wikipedia.csv",
    }

    # Read all
    frames = {}
    for site, path in site_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for site {site}: {path}")
        frames[site] = read_csv(path)

    # Write Excel
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        for site, df in frames.items():
            df.to_excel(writer, index=False, sheet_name=site)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()


