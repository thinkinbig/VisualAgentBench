import csv
from pathlib import Path


CSV_FILENAMES = [
    "site_gitlab.csv",
    "site_map.csv",
    "site_reddit.csv",
    "site_shopping_admin.csv",
    "site_shopping.csv",
    "site_wikipedia.csv",
]


def is_success(value: str) -> bool:
    return str(value).strip().lower() in {"yes", "y", "true", "1"}


def main() -> None:
    templates_dir = Path(__file__).parent / "lite_templates_csv"
    domain_to_counts: dict[str, tuple[int, int]] = {}

    for filename in CSV_FILENAMES:
        csv_path = templates_dir / filename
        if not csv_path.exists():
            print(f"warning: file not found -> {csv_path}")
            continue

        with csv_path.open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                site = (row.get("site", "") or "").strip() or "unknown"
                success_flag = is_success(row.get("success", ""))

                total, successes = domain_to_counts.get(site, (0, 0))
                total += 1
                if success_flag:
                    successes += 1
                domain_to_counts[site] = (total, successes)

    print("domain,success,total,success_rate")
    for site in sorted(domain_to_counts.keys()):
        total, successes = domain_to_counts[site]
        rate = (successes / total) if total else 0.0
        print(f"{site},{successes},{total},{rate:.4f}")


if __name__ == "__main__":
    main()


