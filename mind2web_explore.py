import argparse
import os
from typing import Dict, Tuple

import duckdb
import pandas as pd
from datasets import load_dataset


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_mind2web_train() -> pd.DataFrame:
    dataset = load_dataset("osunlp/Mind2Web", split="train")
    dataframe = dataset.to_pandas()
    return dataframe


def build_tasks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    column_names = [
        "annotation_id",
        "website",
        "domain",
        "subdomain",
        "confirmed_task",
        "action_reprs",
        "actions",
    ]
    tasks_df = df[column_names].copy()
    return tasks_df


def flatten_actions_dataframe(tasks_df: pd.DataFrame) -> pd.DataFrame:
    exploded = (
        tasks_df[["annotation_id", "website", "domain", "confirmed_task", "actions"]]
        .explode("actions")
        .dropna(subset=["actions"])
        .reset_index(drop=True)
    )

    actions_expanded = pd.json_normalize(exploded["actions"])

    # Rename nested operation fields to flat names
    actions_expanded = actions_expanded.rename(
        columns={
            "operation.op": "op",
            "operation.value": "value",
        }
    )

    # Compute candidate counts, default 0 if lists are missing
    def list_length(value) -> int:
        if isinstance(value, list):
            return len(value)
        return 0

    if "pos_candidates" in actions_expanded:
        actions_expanded["num_pos_cands"] = actions_expanded["pos_candidates"].map(list_length)
    else:
        actions_expanded["num_pos_cands"] = 0

    if "neg_candidates" in actions_expanded:
        actions_expanded["num_neg_cands"] = actions_expanded["neg_candidates"].map(list_length)
    else:
        actions_expanded["num_neg_cands"] = 0

    # Drop heavy HTML columns by default to keep outputs lightweight
    columns_to_drop = [
        col
        for col in [
            "raw_html",
            "cleaned_html",
            "pos_candidates",
            "neg_candidates",
            "operation.original_op",
        ]
        if col in actions_expanded.columns
    ]
    actions_expanded = actions_expanded.drop(columns=columns_to_drop, errors="ignore")

    actions_flat = pd.concat(
        [exploded.drop(columns=["actions"]).reset_index(drop=True), actions_expanded.reset_index(drop=True)],
        axis=1,
    )

    # Keep a concise set of columns in a stable order when available
    preferred_order = [
        "annotation_id",
        "website",
        "domain",
        "confirmed_task",
        "action_uid",
        "op",
        "value",
        "num_pos_cands",
        "num_neg_cands",
    ]
    existing = [c for c in preferred_order if c in actions_flat.columns]
    remaining = [c for c in actions_flat.columns if c not in existing]
    actions_flat = actions_flat[existing + remaining]

    return actions_flat


def run_duckdb_stats(tasks_df: pd.DataFrame, actions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    connection = duckdb.connect()
    connection.register("tasks", tasks_df)
    connection.register("actions", actions_df)

    stats: Dict[str, pd.DataFrame] = {}

    stats["domain_task_counts"] = connection.sql(
        """
        select domain, count(*) as num_tasks
        from tasks
        group by domain
        order by num_tasks desc
        """
    ).df()

    stats["op_distribution"] = connection.sql(
        """
        select op, count(*) as num_actions
        from actions
        group by op
        order by num_actions desc
        """
    ).df()

    stats["task_action_lengths"] = connection.sql(
        """
        select annotation_id, count(*) as num_actions
        from actions
        group by annotation_id
        order by num_actions desc
        limit 10
        """
    ).df()

    return stats


def save_outputs(
    out_dir: str,
    tasks_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    stats: Dict[str, pd.DataFrame],
) -> None:
    ensure_directory(out_dir)

    # Data tables
    tasks_path = os.path.join(out_dir, "mind2web_tasks.parquet")
    actions_path = os.path.join(out_dir, "mind2web_actions.parquet")
    tasks_df.to_parquet(tasks_path, index=False)
    actions_df.to_parquet(actions_path, index=False)

    # CSV previews
    ensure_directory(os.path.join(out_dir, "stats"))
    for name, df in stats.items():
        df.to_csv(os.path.join(out_dir, "stats", f"{name}.csv"), index=False)

    # Small sample previews for quick glance
    tasks_df.head(20).to_csv(os.path.join(out_dir, "tasks_preview.csv"), index=False)
    actions_df.head(50).to_csv(os.path.join(out_dir, "actions_preview.csv"), index=False)


def print_human_readable(tasks_df: pd.DataFrame, actions_df: pd.DataFrame, stats: Dict[str, pd.DataFrame]) -> None:
    print("==== Mind2Web (train split) Overview ====")
    print(f"Tasks: {len(tasks_df):,}")
    print(f"Actions (flattened): {len(actions_df):,}")
    print()

    def show(title: str, frame: pd.DataFrame, max_rows: int = 10) -> None:
        print(f"-- {title} --")
        if frame.empty:
            print("<empty>")
        else:
            print(frame.head(max_rows).to_string(index=False))
        print()

    # Print first 5 tasks and actions for quick inspection
    task_cols = ["annotation_id", "website", "domain", "subdomain", "confirmed_task"]
    task_cols = [c for c in task_cols if c in tasks_df.columns]
    show("First 5 tasks", tasks_df[task_cols], max_rows=5)

    action_cols_pref = ["annotation_id", "action_uid", "op", "value", "num_pos_cands", "num_neg_cands"]
    action_cols = [c for c in action_cols_pref if c in actions_df.columns]
    show("First 5 actions", actions_df[action_cols], max_rows=5)

    show("Top domains by number of tasks", stats["domain_task_counts"])
    show("Action type distribution", stats["op_distribution"])
    show("Tasks with most actions", stats["task_action_lengths"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore osunlp/Mind2Web with DuckDB and pandas")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("data", "mind2web"),
        help="Directory to write Parquet files and stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directory(args.out_dir)

    base_df = load_mind2web_train()
    tasks_df = build_tasks_dataframe(base_df)
    actions_df = flatten_actions_dataframe(tasks_df)
    stats = run_duckdb_stats(tasks_df, actions_df)

    save_outputs(args.out_dir, tasks_df, actions_df, stats)
    print_human_readable(tasks_df, actions_df, stats)


if __name__ == "__main__":
    main()


