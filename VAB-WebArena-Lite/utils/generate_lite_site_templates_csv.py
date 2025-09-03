#!/usr/bin/env python3
"""
Generate per-site CSV summaries from the lite WebArena config.

For each site found in test_webarena_lite.raw.json, produce a CSV with columns:
  site,intent_template_id,id_count,eval_types,intent_template,intent,success,agents,remark

If a task lists multiple sites (e.g., ["map", "wikipedia"]) it contributes
to each site's CSV independently.

Usage:
  python generate_lite_site_templates_csv.py \
      --input /home/zeyuli/Code/VisualAgentBench/VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json \
      --outdir /home/zeyuli/Code/VisualAgentBench/VAB-WebArena-Lite/utils/lite_templates_csv

Both flags are optional; sensible defaults are provided for this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set, Optional


@dataclass
class TemplateStats:
    count: int
    example_template: str
    eval_types: Set[str] = field(default_factory=set)
    example_intent: str = ""
    success: bool = False
    agents: Set[str] = field(default_factory=set)
    remark: str = ""


def load_tasks(json_path: Path) -> List[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array of tasks")
    return data


def iter_site_template_records(tasks: Iterable[dict]) -> Iterable[Tuple[str, int, str, Set[str], str, bool, Set[str]]]:
    for task in tasks:
        sites = task.get("sites") or []
        template_id = task.get("intent_template_id")
        template_str = task.get("intent_template", "")
        intent_str = task.get("intent", "")
        tags = task.get("tags") or []
        success = bool(tags)
        agents = {str(t) for t in tags} if isinstance(tags, list) else set()
        eval_types = set()
        eval_block = task.get("eval") or {}
        if isinstance(eval_block, dict):
            raw_types = eval_block.get("eval_types") or []
            if isinstance(raw_types, list):
                eval_types = {str(t) for t in raw_types}

        # Validate essential fields
        if not sites or template_id is None:
            continue

        for site in sites:
            if not isinstance(site, str):
                continue
            yield site, int(template_id), template_str, eval_types, intent_str, success, agents


def aggregate_by_site(records: Iterable[Tuple[str, int, str, Set[str], str, bool, Set[str]]]) -> Dict[str, Dict[int, TemplateStats]]:
    site_to_id_counter: Dict[str, Counter] = defaultdict(Counter)
    site_to_id_templates: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    site_to_id_evaltypes: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
    site_to_id_intents: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    site_to_id_success: Dict[str, Dict[int, List[bool]]] = defaultdict(lambda: defaultdict(list))
    site_to_id_agents: Dict[str, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for site, template_id, template_str, eval_types, intent_str, success, agents in records:
        site_to_id_counter[site][template_id] += 1
        if template_str:
            site_to_id_templates[site][template_id][template_str] += 1
        if eval_types:
            site_to_id_evaltypes[site][template_id].update(eval_types)
        if intent_str:
            site_to_id_intents[site][template_id][intent_str] += 1
        site_to_id_success[site][template_id].append(bool(success))
        if agents:
            site_to_id_agents[site][template_id].update(agents)

    result: Dict[str, Dict[int, TemplateStats]] = {}
    for site, id_counter in site_to_id_counter.items():
        per_site: Dict[int, TemplateStats] = {}
        for template_id, count in id_counter.items():
            # Choose the most frequent template string observed for this id
            example_template_counter = site_to_id_templates[site].get(template_id, Counter())
            if example_template_counter:
                example_template = example_template_counter.most_common(1)[0][0]
            else:
                example_template = ""
            example_intent_counter = site_to_id_intents[site].get(template_id, Counter())
            if example_intent_counter:
                example_intent = example_intent_counter.most_common(1)[0][0]
            else:
                example_intent = ""
            success_vals = site_to_id_success[site].get(template_id, [])
            success_flag = any(success_vals) if success_vals else False
            per_site[template_id] = TemplateStats(
                count=count,
                example_template=example_template,
                eval_types=site_to_id_evaltypes[site].get(template_id, set()),
                example_intent=example_intent,
                success=success_flag,
                agents=site_to_id_agents[site].get(template_id, set()),
            )
        result[site] = per_site
    return result


def write_csvs(aggregated: Dict[str, Dict[int, TemplateStats]], outdir: Path) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for site, id_to_stats in aggregated.items():
        # Sort by count desc, then template_id asc for readability
        rows = sorted(
            (
                (
                    site,
                    template_id,
                    stats.count,
                    "|".join(sorted(stats.eval_types)) if stats.eval_types else "",
                    stats.example_template,
                    stats.example_intent,
                    "yes" if stats.success else "no",
                    "|".join(sorted(stats.agents)) if stats.agents else "",
                    stats.remark,
                )
                for template_id, stats in id_to_stats.items()
            ),
            key=lambda r: (-r[2], r[1]),
        )

        out_path = outdir / f"site_{site}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["site", "intent_template_id", "id_count", "eval_types", "intent_template", "intent", "success", "agents", "remark"])
            for row in rows:
                writer.writerow(row)
        written.append(out_path)
    return written


def load_remarks(path: Optional[Path]) -> Tuple[Dict[int, str], Dict[str, str]]:
    id_to_remark: Dict[int, str] = {}
    tpl_to_remark: Dict[str, str] = {}
    if not path:
        return id_to_remark, tpl_to_remark

    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        id_to_remark[int(k)] = str(v)
                    except Exception:
                        tpl_to_remark[str(k)] = str(v)
            elif isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    _r = row.get("remark")
                    remark = "" if _r is None else str(_r)
                    if "intent_template_id" in row and row["intent_template_id"] is not None:
                        try:
                            id_to_remark[int(row["intent_template_id"])] = remark
                            continue
                        except Exception:
                            pass
                    if "intent_template" in row and row["intent_template"]:
                        tpl_to_remark[str(row["intent_template"])] = remark
        else:
            # Assume CSV with headers
            reader = csv.DictReader(text.splitlines())
            for row in reader:
                if not row:
                    continue
                _r = row.get("remark") or row.get("备注")
                remark = "" if _r is None else str(_r)
                val = row.get("intent_template_id")
                if val not in (None, ""):
                    try:
                        id_to_remark[int(val)] = remark
                        continue
                    except Exception:
                        pass
                tpl = row.get("intent_template")
                if tpl:
                    tpl_to_remark[str(tpl)] = remark
    except Exception:
        # If remarks parsing fails, just return empty mappings
        return id_to_remark, tpl_to_remark

    return id_to_remark, tpl_to_remark


def parse_args() -> argparse.Namespace:
    default_input = Path(
        "/home/zeyuli/Code/VisualAgentBench/VAB-WebArena-Lite/config_files/wa/test_webarena_lite.raw.json"
    )
    default_outdir = Path(
        "/home/zeyuli/Code/VisualAgentBench/VAB-WebArena-Lite/utils/lite_templates_csv"
    )

    parser = argparse.ArgumentParser(description="Generate per-site CSVs for lite templates")
    parser.add_argument("--input", type=Path, default=default_input, help="Path to lite JSON file")
    parser.add_argument("--outdir", type=Path, default=default_outdir, help="Output directory for CSV files")
    parser.add_argument(
        "--remarks",
        type=Path,
        default=None,
        help=(
            "Optional path to remarks mapping. CSV with columns 'intent_template_id' and 'remark' or "
            "'intent_template' and 'remark'; JSON accepted as {id: remark} or list of objects."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.input)
    records = iter_site_template_records(tasks)
    aggregated = aggregate_by_site(records)
    # Apply optional remarks
    id_to_remark, tpl_to_remark = load_remarks(getattr(args, "remarks", None))
    for site_map in aggregated.values():
        for template_id, stats in site_map.items():
            remark = id_to_remark.get(template_id)
            if not remark and stats.example_template:
                remark = tpl_to_remark.get(stats.example_template)
            stats.remark = remark or ""

    written_paths = write_csvs(aggregated, args.outdir)

    # Brief summary to stdout for quick confirmation
    print("Generated CSVs:")
    for p in sorted(written_paths):
        print(f"- {p}")


if __name__ == "__main__":
    main()


