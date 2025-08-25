"""
Bucketing utilities: assign records to 12 buckets = 3 (domain tertiles) x 4 (relative position quartiles).

Functions:
- load_domain_group_map(tertiles_csv, samples)
- compute_task_minmax(samples, task_id)
- rel_bin(value)
- safe_int(value)
- assign_bucket(next_record, samples, tertiles_csv)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, bool):
            return default
        return int(str(x).strip())
    except Exception:
        return default


def rel_bin(rel: float) -> str:
    if rel < 0.25:
        return "Q1_0-25%"
    if rel < 0.50:
        return "Q2_25-50%"
    if rel < 0.75:
        return "Q3_50-75%"
    return "Q4_75-100%"


def compute_task_minmax(samples: List[Dict[str, Any]], task_id: str) -> Tuple[int, int]:
    lo = 10**9
    hi = -10**9
    found = False
    for r in samples:
        if str(r.get("task_id", "")) != str(task_id):
            continue
        found = True
        sid = safe_int(r.get("step_id", 0), 0)
        if sid < lo:
            lo = sid
        if sid > hi:
            hi = sid
    if not found:
        return (0, 0)
    return (lo, hi)


def load_domain_group_map(tertiles_csv: Path, samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """Load domain tertiles mapping if available; otherwise compute from samples."""
    mapping: Dict[str, str] = {}
    if tertiles_csv and tertiles_csv.exists():
        import csv
        with open(tertiles_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                website = str(row.get("website_name", "")).strip()
                group = str(row.get("group", "")).strip()
                if website:
                    mapping[website] = group or "G3_Low"
        if mapping:
            return mapping
    # Fallback compute tertiles by domain counts
    counts = Counter(str(r.get("website_name", "unknown")).strip() or "unknown" for r in samples)
    sorted_domains = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    n = len(sorted_domains)
    if n == 0:
        return mapping
    k, r = divmod(n, 3)
    sizes = [k + (1 if i < r else 0) for i in range(3)]
    labels = ["G1_High", "G2_Mid", "G3_Low"]
    idx = 0
    for gi, sz in enumerate(sizes):
        for _ in range(sz):
            if idx >= n:
                break
            website, _cnt = sorted_domains[idx]
            mapping[website] = labels[gi]
            idx += 1
    return mapping


def assign_bucket(next_record: Dict[str, Any], samples: List[Dict[str, Any]], tertiles_csv: Path | None) -> Dict[str, str]:
    """Return bucket fields: domain_group, rel_bin, bucket for the next_record."""
    task_id = str(next_record.get("task_id", ""))
    website_name = str(next_record.get("website_name", "unknown")).strip() or "unknown"
    mapping = load_domain_group_map(tertiles_csv or Path(), samples)
    domain_group = mapping.get(website_name, "G3_Low")

    lo, hi = compute_task_minmax(samples, task_id)
    step_next = safe_int(next_record.get("step_id", 0))
    lo2 = min(lo, step_next)
    hi2 = max(hi, step_next)
    denom = max(1, hi2 - lo2)
    rel = 0.0 if hi2 == lo2 else (step_next - lo2) / denom
    rb = rel_bin(rel)

    return {
        "domain_group": domain_group,
        "rel_bin": rb,
        "bucket": f"{domain_group}|{rb}",
    }


__all__ = [
    "safe_int",
    "rel_bin",
    "compute_task_minmax",
    "load_domain_group_map",
    "assign_bucket",
]


