#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def collect_old_ids_from_manifest(manifest_path: Path) -> Set[int]:
    data = load_json(manifest_path)
    files: List[str] = data.get("matched_files", [])
    ids: Set[int] = set()
    for name in files:
        try:
            ids.add(int(name.split(".")[0]))
        except Exception:
            continue
    return ids


def collect_old_ids_from_success_list(success_list_path: Path) -> Set[int]:
    items = load_json(success_list_path)
    ids: Set[int] = set()
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                old_id = it.get("old_task_id")
                if isinstance(old_id, int):
                    ids.add(int(old_id))
    return ids


def build_oldid_index(tasks: List[Dict[str, Any]]) -> Dict[int, int]:
    index: Dict[int, int] = {}
    for i, t in enumerate(tasks):
        old_id = t.get("old_task_id")
        if isinstance(old_id, int):
            index[old_id] = i
    return index


def ensure_sources_learn_by_interact(task: Dict[str, Any]) -> bool:
    sources = task.get("sources")
    changed = False
    if sources is None:
        task["sources"] = ["learn_by_interact"]
        return True
    if not isinstance(sources, list):
        task["sources"] = ["learn_by_interact"]
        return True
    if "learn_by_interact" not in sources:
        sources.append("learn_by_interact")
        changed = True
    # Deduplicate while preserving order
    seen: Set[str] = set()
    deduped: List[str] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    task["sources"] = deduped
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill webpilot JSON with filtered tasks and update sources to include 'webarena'")
    parser.add_argument("--webpilot", required=True, type=Path, help="Path to test_webarena_lite_webpilot.raw.json (will be updated)")
    parser.add_argument("--base", required=True, type=Path, help="Path to test_webarena_lite.raw.json (source of full task entries)")
    parser.add_argument("--manifest", required=False, type=Path, help="Path to manifest.json from wa_lite_subset_oldid")
    parser.add_argument("--success_list", required=False, type=Path, help="Path to webpilot_sucess.json (list of {task_id, old_task_id})")
    args = parser.parse_args()

    webpilot_path: Path = args.webpilot
    base_path: Path = args.base

    webpilot_tasks: List[Dict[str, Any]] = load_json(webpilot_path)
    base_tasks: List[Dict[str, Any]] = load_json(base_path)

    old_ids: Set[int] = set()
    if args.success_list and args.success_list.exists():
        old_ids = collect_old_ids_from_success_list(args.success_list)
    elif args.manifest and args.manifest.exists():
        old_ids = collect_old_ids_from_manifest(args.manifest)
    else:
        raise FileNotFoundError("Provide --success_list or --manifest")

    base_index: Dict[int, int] = build_oldid_index(base_tasks)
    web_index: Dict[int, int] = build_oldid_index(webpilot_tasks)

    updated_count = 0
    appended_count = 0
    missing_in_base: List[int] = []

    for old_id in sorted(old_ids):
        base_idx = base_index.get(old_id)
        if base_idx is None:
            missing_in_base.append(old_id)
            continue
        if old_id in web_index:
            wi = web_index[old_id]
            if ensure_sources_learn_by_interact(webpilot_tasks[wi]):
                updated_count += 1
        else:
            new_task = dict(base_tasks[base_idx])
            new_task_sources = new_task.get("sources")
            if isinstance(new_task_sources, list):
                if "learn_by_interact" not in new_task_sources:
                    new_task_sources.append("learn_by_interact")
                seen: Set[str] = set()
                deduped: List[str] = []
                for s in new_task_sources:
                    if s not in seen:
                        seen.add(s)
                        deduped.append(s)
                new_task["sources"] = deduped
            else:
                new_task["sources"] = ["learn_by_interact"]
            webpilot_tasks.append(new_task)
            appended_count += 1
            web_index[old_id] = len(webpilot_tasks) - 1

    save_json(webpilot_path, webpilot_tasks)

    print(f"Updated sources for existing tasks: {updated_count}")
    print(f"Appended new tasks from base: {appended_count}")
    if missing_in_base:
        print(f"Warning: {len(missing_in_base)} old_task_id not found in base config (first 20): {missing_in_base[:20]}")


if __name__ == "__main__":
    main()
