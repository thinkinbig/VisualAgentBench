#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Set, List, Tuple


def collect_task_ids_from_config(config_path: Path) -> Set[int]:
    with config_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    ids: Set[int] = set()
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            old_tid = item.get("old_task_id")
            if isinstance(old_tid, int):
                ids.add(int(old_tid))
    elif isinstance(data, dict):
        # Recursively traverse; only collect old_task_id
        def walk(obj):
            if isinstance(obj, dict):
                old_tid = obj.get("old_task_id")
                if isinstance(old_tid, int):
                    ids.add(int(old_tid))
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
        walk(data)
    return ids


def find_matching_files(source_dir: Path, task_ids: Set[int]) -> Tuple[List[Path], List[int]]:
    matched_files: List[Path] = []
    missing_ids: List[int] = []
    existing_ids: Set[int] = set()

    digit_json = re.compile(r"^(\d+)\.json$")
    for entry in sorted(source_dir.iterdir()):
        if not entry.is_file():
            continue
        m = digit_json.match(entry.name)
        if not m:
            continue
        file_id = int(m.group(1))
        if file_id in task_ids:
            matched_files.append(entry)
            existing_ids.add(file_id)

    for tid in sorted(task_ids):
        if tid not in existing_ids:
            missing_ids.append(tid)

    return matched_files, missing_ids


def write_manifest(output_dir: Path, matched_files: List[Path], missing_ids: List[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "matched_count": len(matched_files),
        "missing_count": len(missing_ids),
        "matched_files": [p.name for p in matched_files],
        "missing_task_ids": missing_ids,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8'
    )


def copy_files(matched_files: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in matched_files:
        dst = output_dir / src.name
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WebArena Lite subset from success trajectories")
    parser.add_argument("--config", required=True, type=Path, help="Path to test_webarena_lite.raw.json")
    parser.add_argument("--source", required=True, type=Path, help="Path to success trajectories directory containing <task_id>.json files")
    parser.add_argument("--output", required=True, type=Path, help="Directory to write the subset trajectories")
    args = parser.parse_args()

    config_path: Path = args.config
    source_dir: Path = args.source
    output_dir: Path = args.output

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")

    task_ids = collect_task_ids_from_config(config_path)
    print(f"Collected {len(task_ids)} lite old_task_id from {config_path}")

    matched_files, missing_ids = find_matching_files(source_dir, task_ids)
    print(f"Found {len(matched_files)} matching success trajectories in {source_dir}")
    if missing_ids:
        print(f"Missing {len(missing_ids)} trajectories (no file present) for task_ids: {missing_ids[:20]}{' ...' if len(missing_ids) > 20 else ''}")

    copy_files(matched_files, output_dir)
    write_manifest(output_dir, matched_files, missing_ids)

    print(f"Copied subset to: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
