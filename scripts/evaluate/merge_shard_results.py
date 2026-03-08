#!/usr/bin/env python3
"""Merge shard-level evaluate.py JSON outputs into one consolidated JSON."""

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _discover_mode_names(results: List[Dict[str, Any]]) -> List[str]:
    names = set()
    for row in results:
        modes = row.get("modes", {})
        if isinstance(modes, dict):
            names.update(modes.keys())
    return sorted(names)


def _compute_mode_results(
    results: List[Dict[str, Any]],
    mode_names: List[str],
) -> Dict[str, Dict[str, Any]]:
    mode_results: Dict[str, Dict[str, Any]] = {}
    for mode in mode_names:
        correct = 0
        total = 0
        for row in results:
            mode_row = row.get("modes", {}).get(mode, {})
            if not isinstance(mode_row, dict):
                continue
            if "error" in mode_row:
                continue
            if "is_correct" not in mode_row:
                continue
            total += 1
            if bool(mode_row.get("is_correct", False)):
                correct += 1
        mode_results[mode] = {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total * 100.0) if total > 0 else 0.0,
        }
    return mode_results


def merge_shards(input_files: List[str], out_file: str) -> None:
    if not input_files:
        raise ValueError("No input files provided.")

    shard_data = []
    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON root in: {path}")
        shard_data.append((path, data))

    first_summary = deepcopy(shard_data[0][1].get("summary", {}))

    merged_results: List[Dict[str, Any]] = []
    seen_qids = set()
    duplicate_count = 0
    for _, data in shard_data:
        rows = data.get("results", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            qid = row.get("question_id")
            qid_key = _safe_int(qid)
            if qid_key in seen_qids:
                duplicate_count += 1
                continue
            seen_qids.add(qid_key)
            merged_results.append(row)

    merged_results.sort(key=lambda r: _safe_int(r.get("question_id")))

    # Recompute accuracy-like mode metrics from merged rows.
    mode_names = _discover_mode_names(merged_results)
    mode_results = _compute_mode_results(merged_results, mode_names)

    summary = first_summary if isinstance(first_summary, dict) else {}
    config = deepcopy(summary.get("config", {})) if isinstance(summary.get("config", {}), dict) else {}
    config["total_evaluated"] = len(merged_results)
    config["merged_from_shards"] = len(input_files)
    config["duplicate_question_rows_dropped"] = duplicate_count
    summary["config"] = config

    total_seconds = 0.0
    for _, data in shard_data:
        t = data.get("summary", {}).get("timing", {}).get("total_seconds", 0.0)
        try:
            total_seconds += float(t)
        except Exception:
            pass

    summary["timing"] = {
        "total_seconds": total_seconds,
        "avg_per_question": (total_seconds / len(merged_results)) if merged_results else 0.0,
        "questions_per_minute": ((len(merged_results) / total_seconds) * 60.0) if total_seconds > 0 else 0.0,
    }
    summary["mode_results"] = mode_results
    summary["timestamp"] = datetime.now().isoformat()
    summary["merge_info"] = {
        "input_files": input_files,
        "note": (
            "mode_results/timing are recomputed from merged rows; "
            "other task-specific aggregates may remain from shard metadata."
        ),
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "results": merged_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Merged {len(input_files)} shard files")
    print(f"Total rows: {len(merged_results)} (dropped duplicates: {duplicate_count})")
    print(f"Saved: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    merge_shards(args.inputs, args.out)


if __name__ == "__main__":
    main()

