#!/usr/bin/env python3
"""
Orchestrate GRPO checkpoint evaluation on MedQA (evaluate.py backend).

Thin wrapper: discovers checkpoints, assigns GPUs, and launches
evaluate.run_batch_phased_evaluation() in a separate process
per GPU. Each GPU processes checkpoints sequentially.

Supports TWO modes (auto-detected from hyperparameters.json):
  rewriter   — checkpoint used in Phase 2 (rewrite queries)
  hypothesis — checkpoint used in Phase 1 (generate diagnostic plan)

Only one model on GPU at a time → full vLLM batch throughput.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ── Project paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

for p in [
    SCRIPT_DIR,
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "MIRAGE"),
    os.path.join(PROJECT_ROOT, "MIRAGE", "MedRAG"),
    os.path.join(PROJECT_ROOT, "MIRAGE", "MedRAG", "src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helpers ──────────────────────────────────────────────────────────────────

def discover_checkpoints(training_dir: str, every_n: int = 1) -> list:
    ckpt_dirs = []
    for name in os.listdir(training_dir):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m and os.path.isdir(os.path.join(training_dir, name)):
            step = int(m.group(1))
            ckpt_dirs.append((step, os.path.join(training_dir, name)))
    ckpt_dirs.sort(key=lambda x: x[0])
    if every_n > 1:
        ckpt_dirs = [(s, p) for s, p in ckpt_dirs if s % every_n == 0]
    return ckpt_dirs


def is_valid_model_checkpoint_dir(path: str) -> bool:
    """Returns True if checkpoint dir looks loadable by HF/vLLM."""
    if not os.path.isdir(path):
        return False
    names = set(os.listdir(path))
    return any(
        key in names for key in ["config.json", "params.json", "adapter_config.json"]
    )


def read_hyperparameters(training_dir: str) -> dict:
    with open(os.path.join(training_dir, "hyperparameters.json")) as f:
        return json.load(f)


def detect_mode(hparams: dict) -> str:
    """Auto-detect evaluation mode from hyperparameters.
    Returns 'hypothesis' if adapter_out_dir contains 'hypothesis',
    otherwise 'rewriter'.
    """
    adapter_dir = hparams.get("adapter_out_dir", "")
    if "hypothesis" in adapter_dir.lower():
        return "hypothesis"
    return "rewriter"


def is_checkpoint_done(results_dir: str, step: int) -> bool:
    d = os.path.join(results_dir, f"checkpoint-{step}")
    if os.path.isdir(d):
        return any(f.endswith(".json") for f in os.listdir(d))
    return False


def _resolve_gpu_mem(desired: float, tag: str, margin: float = 0.95) -> float:
    """
    Clip desired vLLM gpu memory utilization by currently available memory.
    This avoids immediate startup failures like:
    free_mem < desired_util * total_mem.
    """
    try:
        import torch
    except Exception:
        print(f"{tag} torch is unavailable, using requested gpu_mem={desired:.2f}")
        return desired

    if not torch.cuda.is_available():
        return desired

    try:
        free, total = torch.cuda.mem_get_info(0)
        if total <= 0:
            return desired
        free_ratio = free / total
        safe = min(desired, free_ratio * margin)
        safe = min(max(safe, 0.1), 1.0)
        if safe < desired:
            free_gib = free / (1024 ** 3)
            total_gib = total / (1024 ** 3)
            print(
                f"{tag} GPU memory utilization auto-tuned: {desired:.2f} -> {safe:.2f}"
                f" (free {free_gib:.2f}/{total_gib:.2f} GiB)"
            )
        return safe
    except Exception as e:
        print(f"{tag} Could not query GPU memory ({e}); using gpu_mem={desired:.2f}")
        return desired


# ── Per-GPU Worker Process ──────────────────────────────────────────────────

def gpu_worker(
    gpu_id: int,
    assigned: List[Tuple[int, str]],
    base_model: str,
    results_base: str,
    max_questions: int,
    total_docs: int,
    gpu_mem: float,
    max_model_len: int,
    max_tokens: int,
    mode: str = "rewriter",
    hypothesis_checkpoint: Optional[str] = None,
    rewriter_checkpoint: Optional[str] = None,
    generator_checkpoint: Optional[str] = None,
    hypothesis_prompt: str = "v1",
    rewriting_prompt: str = "v1",
    generator_prompt: str = "v1",
    vllm_tensor_parallel_size: int = 1,
):
    """
    Worker process for one GPU.
    Runs run_batch_phased_evaluation() for each assigned checkpoint.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tag = f"[GPU {gpu_id}]"
    print(f"{tag} Worker started (PID {os.getpid()}, mode={mode})")

    # Late import (after CUDA_VISIBLE_DEVICES is set)
    from evaluate import run_batch_phased_evaluation

    effective_gpu_mem = _resolve_gpu_mem(gpu_mem, tag)

    for step, ckpt_path in assigned:
        out_dir = os.path.join(results_base, f"checkpoint-{step}")
        if is_checkpoint_done(results_base, step):
            print(f"{tag} Checkpoint {step}: already done, skipping")
            continue

        print(f"\n{tag} ═══ Checkpoint {step} ═══")
        t0 = time.time()
        try:
            # Build kwargs based on mode: checkpoint goes to the right module
            kwargs = dict(
                base_model=base_model,
                max_questions=max_questions,
                total_docs=total_docs,
                gpu_mem=effective_gpu_mem,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
                max_model_len=max_model_len,
                max_tokens=max_tokens,
                output_dir=out_dir,
                hypothesis_prompt=hypothesis_prompt,
                rewriting_prompt=rewriting_prompt,
                generator_prompt=generator_prompt,
            )
            if mode == "hypothesis":
                kwargs["hypothesis_checkpoint"] = ckpt_path
                # Pass additional fixed checkpoints if specified
                if rewriter_checkpoint:
                    kwargs["rewriter_checkpoint"] = rewriter_checkpoint
                if generator_checkpoint:
                    kwargs["generator_checkpoint"] = generator_checkpoint
            else:
                kwargs["rewriter_checkpoint"] = ckpt_path
                # Pass additional fixed checkpoints if specified
                if hypothesis_checkpoint:
                    kwargs["hypothesis_checkpoint"] = hypothesis_checkpoint
                if generator_checkpoint:
                    kwargs["generator_checkpoint"] = generator_checkpoint

            run_batch_phased_evaluation(**kwargs)
            elapsed = time.time() - t0
            print(f"{tag} ✓ Checkpoint {step} done in {elapsed:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{tag} ✗ Checkpoint {step} failed: {e}")

    print(f"\n{tag} Worker done.")


# ── Summary ──────────────────────────────────────────────────────────────────

def generate_summary(training_dir, checkpoints, results_base, total_time=None):
    os.makedirs(results_base, exist_ok=True)
    all_res = []
    for step, _ in checkpoints:
        d = os.path.join(results_base, f"checkpoint-{step}")
        if not os.path.isdir(d):
            continue
        jf = sorted([f for f in os.listdir(d) if f.endswith(".json")])
        if not jf:
            continue
        with open(os.path.join(d, jf[-1])) as f:
            data = json.load(f)
        s = data.get("summary", {})
        mode_results = s.get("mode_results", {}) or {}
        if mode_results:
            mode_key = next(iter(mode_results.keys()))
            mr = mode_results.get(mode_key, {})
        else:
            mode_key = "unknown"
            mr = {}
        all_res.append({
            "step": step,
            "mode": mode_key,
            "accuracy": mr.get("accuracy", 0),
            "correct": mr.get("correct", 0),
            "total": mr.get("total", 0),
            "time_seconds": s.get("timing", {}).get("total_seconds", 0),
        })
    if not all_res:
        print("No results to summarize.")
        return
    all_res.sort(key=lambda x: x["step"])

    csv_path = os.path.join(results_base, "checkpoint_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "step", "mode", "accuracy", "correct", "total", "time_seconds"])
        w.writeheader()
        w.writerows(all_res)

    json_path = os.path.join(results_base, "checkpoint_summary.json")
    with open(json_path, "w") as f:
        json.dump({"training_dir": training_dir,
                    "total_checkpoints": len(all_res),
                    "total_time_seconds": total_time,
                    "timestamp": datetime.now().isoformat(),
                    "results": all_res}, f, indent=2)

    best = max(all_res, key=lambda x: x["accuracy"])
    print(f"\n{'='*70}\nCHECKPOINT EVALUATION SUMMARY\n{'='*70}")
    print(f"{'Step':>8}  {'Accuracy':>10}  {'Correct':>8}  "
          f"{'Total':>6}  {'Time':>8}")
    print("-" * 70)
    for r in all_res:
        m = " ★" if r["step"] == best["step"] else ""
        print(f"  {r['step']:>6}  {r['accuracy']:>9.1f}%  "
              f"{r['correct']:>7}/{r['total']:<6}  "
              f"{r['time_seconds']:.0f}s{m}")
    print(f"\n  Best: checkpoint-{best['step']} ({best['accuracy']:.1f}%)")
    if total_time:
        print(f"  Wall-clock: {total_time/60:.1f} min")
    print(f"  CSV:  {csv_path}\n  JSON: {json_path}\n{'='*70}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate GRPO checkpoint evaluation on MedQA")
    parser.add_argument("training_dir",
                        help="Training output dir with checkpoint-* dirs")
    parser.add_argument("--mode", choices=["rewriter", "hypothesis"],
                        default=None,
                        help="Evaluation mode (default: auto-detect)")
    parser.add_argument("--gpus", default="3,4,5",
                        help="Comma-separated GPU IDs (default: 3,4,5)")
    parser.add_argument("--max-questions", "-n", type=int, default=1273)
    parser.add_argument("--every-n", type=int, default=1,
                        help="Evaluate every N-th step (default: 1 = all)")
    parser.add_argument("--total-docs", type=int, default=15)
    parser.add_argument("--gpu-mem", type=float, default=0.9,
                        help="GPU memory utilization (default: 0.9)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--hypothesis-prompt", type=str, default=None,
                        choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'])
    parser.add_argument("--rewriting-prompt", type=str, default=None,
                        choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'])
    parser.add_argument("--generator-prompt", type=str, default=None,
                        choices=['v1', 'v2'])
    parser.add_argument("--results-subdir", type=str, default="results_v2",
                        help="Results directory name under training_dir (default: results_v2)")
    # Per-module checkpoint overrides (fixed, not iterated)
    parser.add_argument("--hypothesis-checkpoint", type=str, default=None,
                        help="Fixed hypothesis checkpoint (when mode=rewriter)")
    parser.add_argument("--rewriter-checkpoint", type=str, default=None,
                        help="Fixed rewriter checkpoint (when mode=hypothesis)")
    parser.add_argument("--generator-checkpoint", type=str, default=None,
                        help="Fixed generator checkpoint")
    args = parser.parse_args()

    training_dir = os.path.abspath(args.training_dir)
    hparams = read_hyperparameters(training_dir)
    base_model = hparams["base_model"]
    mode = args.mode if args.mode else detect_mode(hparams)
    hypothesis_prompt = args.hypothesis_prompt or hparams.get("hypothesis_prompt", "v1")
    rewriting_prompt = args.rewriting_prompt or hparams.get("rewriting_prompt", "v1")
    generator_prompt = args.generator_prompt or hparams.get("generator_prompt", "v1")

    print(f"Base model: {base_model}")
    print(f"Mode: {mode}")
    print(f"Prompts: hypothesis={hypothesis_prompt}, rewriting={rewriting_prompt}, generator={generator_prompt}")

    checkpoints = discover_checkpoints(training_dir, args.every_n)
    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)

    valid, invalid = [], []
    for step, path in checkpoints:
        if is_valid_model_checkpoint_dir(path):
            valid.append((step, path))
        else:
            invalid.append((step, path))

    if invalid:
        bad_steps = [s for s, _ in invalid]
        print(f"Skipping {len(invalid)} invalid checkpoints (no model files): {bad_steps}")

    if not valid:
        print("No valid checkpoints found.")
        sys.exit(1)

    results_base = os.path.join(training_dir, args.results_subdir)
    pending = [(s, p) for s, p in valid
               if not is_checkpoint_done(results_base, s)]

    print(f"Checkpoints: {len(valid)} valid, "
          f"{len(valid)-len(pending)} done, {len(pending)} pending")
    if not pending:
        generate_summary(training_dir, valid, results_base)
        sys.exit(0)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    n_gpus = len(gpu_ids)
    print(f"GPUs: {gpu_ids} ({n_gpus} parallel)\n")

    # Round-robin assignment
    assignment: Dict[int, list] = {g: [] for g in gpu_ids}
    for idx, ckpt in enumerate(pending):
        assignment[gpu_ids[idx % n_gpus]].append(ckpt)
    for g in gpu_ids:
        steps = [s for s, _ in assignment[g]]
        print(f"  GPU {g}: {len(steps)} checkpoints → {steps}")

    # One process per GPU
    processes = []
    t0 = time.time()
    for gpu_id in gpu_ids:
        if not assignment[gpu_id]:
            continue
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, assignment[gpu_id], base_model, results_base,
                  args.max_questions, args.total_docs, args.gpu_mem,
                  args.max_model_len, args.max_tokens,
                  mode, args.hypothesis_checkpoint,
                  args.rewriter_checkpoint, args.generator_checkpoint,
                  hypothesis_prompt, rewriting_prompt, generator_prompt,
                  args.vllm_tensor_parallel_size),
        )
        processes.append(p)
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
        for p in processes:
            p.terminate()

    total_time = time.time() - t0
    generate_summary(training_dir, valid, results_base, total_time)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
