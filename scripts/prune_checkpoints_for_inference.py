#!/usr/bin/env python3
"""
Prune old checkpoints so that only inference-ready files remain.

For each directory containing `checkpoint-<step>` folders (for example one training run),
the checkpoint with the highest step is kept untouched.
For older checkpoints, files matching inference patterns are kept while train artifacts
are removed.

Default behavior is safe: without --apply, the script only prints the planned actions.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


STEP_RE = re.compile(r"^checkpoint-(\d+)$")

INFERENCE_KEEP_PATTERNS = (
    "config.json",
    "generation_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin.index.json",
    "adapter_model.safetensors",
    "adapter_model-*.safetensors",
    "adapter_config.json",
    "README.md",
)

TRAIN_DELETE_PATTERNS = (
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "rng_state_*.pth",
)


@dataclass
class CheckpointPlan:
    checkpoint: Path
    step: int
    delete_files: List[Path]
    keep_files: List[Path]
    unknown_files: List[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete train artifacts from checkpoint directories except the latest step, "
            "while keeping inference files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs"),
        help="Root directory to search for checkpoint-* folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files (without this flag, only show planned actions).",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help=(
            "Delete files that are neither known inference files nor known train files."
        ),
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help=(
            "Extra file glob patterns to keep (can be passed multiple times). "
            "Example: --keep '*.json'"
        ),
    )
    parser.add_argument(
        "--delete",
        action="append",
        default=[],
        help=(
            "Extra file glob patterns to delete (can be passed multiple times). "
            "Example: --delete '*_state.bin'"
        ),
    )
    return parser.parse_args()


def match_pattern(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def discover_checkpoint_dirs(root: Path) -> dict[Path, List[Tuple[int, Path]]]:
    groups: dict[Path, List[Tuple[int, Path]]] = defaultdict(list)
    for d in root.rglob("checkpoint-*"):
        if not d.is_dir():
            continue
        m = STEP_RE.match(d.name)
        if not m:
            continue
        groups[d.parent].append((int(m.group(1)), d))
    return groups


def make_plan(
    root: Path,
    keep_patterns: Iterable[str],
    delete_patterns: Iterable[str],
    aggressive: bool,
) -> List[CheckpointPlan]:
    final_keep = tuple(keep_patterns)
    final_delete = tuple(delete_patterns)
    plans: List[CheckpointPlan] = []

    groups = discover_checkpoint_dirs(root)
    for parent, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(items) <= 1:
            continue
        items.sort(key=lambda x: x[0])
        latest_step = items[-1][0]

        for step, ckpt in items:
            if step == latest_step:
                continue

            delete_files: List[Path] = []
            keep_files: List[Path] = []
            unknown_files: List[Path] = []

            for item in sorted(ckpt.iterdir()):
                if not item.is_file():
                    continue
                name = item.name
                if match_pattern(name, final_delete) or match_pattern(name, TRAIN_DELETE_PATTERNS):
                    delete_files.append(item)
                elif match_pattern(name, final_keep) or match_pattern(name, INFERENCE_KEEP_PATTERNS):
                    keep_files.append(item)
                elif aggressive:
                    delete_files.append(item)
                else:
                    unknown_files.append(item)

            plans.append(
                CheckpointPlan(
                    checkpoint=ckpt,
                    step=step,
                    delete_files=delete_files,
                    keep_files=keep_files,
                    unknown_files=unknown_files,
                )
            )
    return plans


def prompt_yes_no(msg: str) -> bool:
    while True:
        ans = input(f"{msg} [y/N]: ").strip().lower()
        if ans in ("", "n", "no"):
            return False
        if ans in ("y", "yes"):
            return True
        print("  Enter y or n.")


def show_plan(plans: List[CheckpointPlan]) -> None:
    if not plans:
        print("No checkpoints were selected for cleanup.")
        return

    print("\nPlanned actions:")
    for p in plans:
        print(f"\n{p.checkpoint} (step={p.step})")
        print(f"  delete: {len(p.delete_files)}")
        for f in p.delete_files:
            print(f"   - {f.name}")
        if p.unknown_files:
            print(f"  unknown (kept): {len(p.unknown_files)}")
            for f in p.unknown_files:
                print(f"   - {f.name}")
        print(f"  keep (inference): {len(p.keep_files)}")
        for f in p.keep_files:
            print(f"   - {f.name}")


def apply_plan(plans: List[CheckpointPlan]) -> None:
    total = sum(len(p.delete_files) for p in plans)
    active_plans = [p for p in plans if p.delete_files]
    total_active_checkpoints = len(active_plans)
    total_unknown = sum(len(p.unknown_files) for p in active_plans)
    print(f"\nTotal files scheduled for deletion: {total} "
          f"(across {total_active_checkpoints} checkpoints)")

    if not prompt_yes_no("Apply the planned deletions now?"):
        print("Aborted.")
        return

    if total_unknown:
        print(f"\nUnknown files preserved: {total_unknown}")
        for p in active_plans:
            if not p.unknown_files:
                continue
            print(f"\n{p.checkpoint} (step={p.step})")
            for f in p.unknown_files:
                print(f"  - {f.name}")

    for p in active_plans:
        for f in p.delete_files:
            try:
                f.unlink()
                print(f"  deleted: {f}")
            except Exception as e:
                print(f"  failed: {f} ({e})")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"Error: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    keep_patterns = INFERENCE_KEEP_PATTERNS + tuple(args.keep)
    delete_patterns = TRAIN_DELETE_PATTERNS + tuple(args.delete)
    plans = make_plan(
        root=root,
        keep_patterns=keep_patterns,
        delete_patterns=delete_patterns,
        aggressive=args.aggressive,
    )

    show_plan(plans)

    if not args.apply:
        print("\nTip: run with --apply to perform deletions.")
        return

    if not plans:
        return

    apply_plan(plans)


if __name__ == "__main__":
    main()
