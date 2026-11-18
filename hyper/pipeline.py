"""HYPER evaluation pipeline."""
import os
import re
import json
import math
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from hyper.prompts import HYPOTHESIS_PROMPT, QUERY_PROMPT, ANSWER_PROMPT
from hyper.parsing import (
    parse_answer,
    coerce_choice,
    coerce_confidence_level,
    _enrich_plan_with_alternatives,
)
from hyper.inference import _run_phase_in_subprocess

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
EVAL_FIXED_TEMPERATURE = 0.0
EVAL_FIXED_SEED = 42
EVAL_FIXED_DO_SAMPLE = False


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return value or "model"


def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility setup across common libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _to_json_safe(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _to_json_safe(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _serialize_retrieved_docs_for_output(
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep retrieval evidence in output JSON with stable, safe fields."""
    export_keys = [
        "id",
        "doc_id",
        "chunk_id",
        "title",
        "source",
        "content",
        "text",
        "score",
        "fused_score",
        "query_trace",
        "metadata",
    ]
    serialized: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            serialized.append(
                {
                    "id": f"doc_{idx+1}",
                    "content": _to_json_safe(doc),
                }
            )
            continue

        row: Dict[str, Any] = {}
        for key in export_keys:
            if key in doc:
                row[key] = _to_json_safe(doc.get(key))

        if "id" not in row:
            row["id"] = str(
                doc.get("id")
                or doc.get("chunk_id")
                or doc.get("doc_id")
                or f"doc_{idx+1}"
            )
        if "title" not in row and doc.get("title") is not None:
            row["title"] = _to_json_safe(doc.get("title"))

        if "content" not in row:
            if doc.get("content") is not None:
                row["content"] = _to_json_safe(doc.get("content"))
            elif doc.get("text") is not None:
                row["content"] = _to_json_safe(doc.get("text"))
            else:
                row["content"] = ""

        serialized.append(row)
    return serialized


class BenchmarkDataset:
    """benchmark.json dataset loader"""

    def __init__(
        self, benchmark_path: Optional[str] = None, question_dataset: str = "medqa"
    ):
        if benchmark_path is None:
            possible_paths = [
                os.path.join(PROJECT_ROOT, "MIRAGE", "benchmark.json"),
                os.path.join(PROJECT_ROOT, "data", "benchmark.json"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    benchmark_path = path
                    break
            if benchmark_path is None:
                raise FileNotFoundError(
                    "benchmark.json not found; run python scripts/prepare_data.py"
                )

        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

        if question_dataset not in benchmark:
            raise KeyError(
                f"Dataset '{question_dataset}' not found in benchmark.json. "
                f"Available: {sorted(benchmark.keys())}"
            )

        self.question_dataset = question_dataset
        self.dataset = benchmark[question_dataset]
        self.index = sorted(self.dataset.keys())
        print(f"✓ Loaded {len(self)} {question_dataset} questions")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key) -> Dict[str, Any]:
        if isinstance(key, int):
            return self.dataset[self.index[key]]
        elif isinstance(key, slice):
            return [self.__getitem__(i) for i in range(len(self))[key]]
        else:
            raise KeyError(f"Key type {type(key)} not supported")


def _format_list(val):
    """Convert list or string to comma-separated string."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val[:5])
    return str(val)


def resolve_results_output_dir(output_dir, llm_provider, model_name):
    return output_dir or os.path.join(PROJECT_ROOT, "outputs", _slugify(model_name))


def evaluate(
    base_model: str,
    max_questions: int = 1273,
    total_docs: int = 15,
    question_dataset: str = "medqa",
    gpu_mem: float = 0.85,
    max_model_len: int = 8192,
    max_tokens: int = 2048,
    output_dir: Optional[str] = None,
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    vllm_tensor_parallel_size: int = 1,
) -> Dict[str, Any]:
    """Evaluate HYPER with batched hypothesis, query, retrieval, and answer stages."""
    output_dir = resolve_results_output_dir(
        output_dir=output_dir, llm_provider="vllm", model_name=base_model
    )
    set_global_seed(EVAL_FIXED_SEED)
    from hyper.parsing import parse_queries_from_completion, parse_hypothesis_plan

    hypothesis_model = base_model
    rewriter_model = base_model
    generator_model = base_model
    h_prompt = HYPOTHESIS_PROMPT
    r_prompt = QUERY_PROMPT
    g_prompt = ANSWER_PROMPT
    method = "hyper"
    os.makedirs(output_dir, exist_ok=True)
    dataset = BenchmarkDataset(question_dataset=question_dataset)
    capped_total = max(0, min(max_questions, len(dataset)))
    question_ids = list(range(capped_total))
    questions = [dataset[i] for i in question_ids]
    n = len(questions)
    print(f"\n{'=' * 70}")
    print(f"Batch-Phased Evaluation: {n} questions")
    print(f"Question set: {question_dataset}")
    print(f"Method:       {method}")
    print(f"Base model:   {base_model}")
    print(
        f"vllm_tp={vllm_tensor_parallel_size}  gpu_mem={gpu_mem}  max_model_len={max_model_len}  max_tokens={max_tokens}"
    )
    print(f"seed={EVAL_FIXED_SEED}  temperature={EVAL_FIXED_TEMPERATURE}")
    print(f"do_sample={EVAL_FIXED_DO_SAMPLE} (where supported)")
    print(f"{'=' * 70}")
    t_total = time.time()
    print(
        f"\n[Phase 1/4] Generating hypotheses for {n} questions ({os.path.basename(hypothesis_model)}) ..."
    )
    t1 = time.time()
    plan_messages = []
    for qd in questions:
        options_text = "\n".join(
            [f"{k}. {v}" for (k, v) in sorted(qd["options"].items())]
        )
        user = h_prompt["user"].format(question=qd["question"], options=options_text)
        plan_messages.append(
            [
                {"role": "system", "content": h_prompt["system"]},
                {"role": "user", "content": user},
            ]
        )
    plan_texts = _run_phase_in_subprocess(
        hypothesis_model,
        plan_messages,
        gpu_mem,
        max_model_len,
        max_tokens,
        vllm_tensor_parallel_size,
    )
    plans = []
    best_guess_correct = 0
    closest_alternative_correct = 0
    closest_alternative_rescue = 0
    for (idx, text) in enumerate(plan_texts):
        p = parse_hypothesis_plan(text)
        p.setdefault("discriminating_features", [])
        p.setdefault("best_guess", "")
        p.setdefault("best_guess_text", "")
        p.setdefault("reasoning", "")
        p.setdefault("confirming_evidence", [])
        p.setdefault("alternative_if_wrong", "")
        p.setdefault("closest_alternative", "")
        p.setdefault("closest_alternative_text", "")
        p["confidence_level"] = coerce_confidence_level(
            p.get("confidence_level", 2), default=2
        )
        if not p["discriminating_features"]:
            words = re.findall("\\b[A-Za-z]{4,}\\b", questions[idx]["question"])
            p["discriminating_features"] = words[:3] or ["symptom"]
        p = _enrich_plan_with_alternatives(p, text, questions[idx].get("options", {}))
        gold = coerce_choice(
            questions[idx].get("answer_idx", questions[idx].get("answer", ""))
        )
        best_guess = coerce_choice(p.get("best_guess", ""))
        closest_alternative = coerce_choice(
            p.get("closest_alternative", p.get("alternative_if_wrong", ""))
        )
        if best_guess == gold and bool(gold):
            best_guess_correct += 1
        if closest_alternative == gold and bool(gold):
            closest_alternative_correct += 1
            if best_guess != gold:
                closest_alternative_rescue += 1
        plans.append(p)
    print(f"  ✓ Phase 1 done: {len(plans)} hypotheses in {time.time() - t1:.1f}s")
    best_guess_accuracy = best_guess_correct / n * 100 if n > 0 else 0.0
    closest_alternative_accuracy = (
        closest_alternative_correct / n * 100 if n > 0 else 0.0
    )
    top2_accuracy = (
        (best_guess_correct + closest_alternative_rescue) / n * 100 if n > 0 else 0.0
    )
    print(
        f"  ✓ best_guess={best_guess_accuracy:.2f}% | closest_alt={closest_alternative_accuracy:.2f}% | top2={top2_accuracy:.2f}%"
    )
    print(
        f"\n[Phase 2/4] Generating queries for {n} questions ({os.path.basename(rewriter_model)}) ..."
    )
    t2 = time.time()
    rw_messages = []
    for (i, qd) in enumerate(questions):
        p = plans[i]
        options_text = "\n".join(
            [f"{k}. {v}" for (k, v) in sorted(qd["options"].items())]
        )
        bg_letter = (p.get("best_guess", "") or "").strip().upper().rstrip(".")
        bg_text = p.get("best_guess_text") or (
            f"{bg_letter}. {qd['options'].get(bg_letter, bg_letter)}"
            if bg_letter in qd.get("options", {})
            else bg_letter
        )
        alt_letter = (
            (
                p.get("closest_alternative", "")
                or p.get("alternative_if_wrong", "")
                or ""
            )
            .strip()
            .upper()
            .rstrip(".")
        )
        closest_alternative_text = p.get("closest_alternative_text") or ""
        alt_text = closest_alternative_text or (
            f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
            if alt_letter in qd.get("options", {})
            else alt_letter
        )
        user = r_prompt["user"].format(
            question=qd["question"],
            options=options_text,
            best_guess=p.get("best_guess", ""),
            best_guess_text=bg_text,
            reasoning=p.get("reasoning", ""),
            confirming_evidence=_format_list(p.get("confirming_evidence", [])),
            discriminating_features=_format_list(p.get("discriminating_features", [])),
            alternative_if_wrong=p.get("alternative_if_wrong", ""),
            alternative_text=alt_text,
            closest_alternative=p.get("closest_alternative", ""),
            closest_alternative_text=alt_text,
        )
        rw_messages.append(
            [
                {"role": "system", "content": r_prompt["system"]},
                {"role": "user", "content": user},
            ]
        )
    rw_texts = _run_phase_in_subprocess(
        rewriter_model,
        rw_messages,
        gpu_mem,
        max_model_len,
        max_tokens,
        vllm_tensor_parallel_size,
    )
    all_queries = []
    for (i, text) in enumerate(rw_texts):
        queries = parse_queries_from_completion(text)
        if len(queries) < 2:
            p = plans[i]
            bg = p.get("best_guess", "")
            cf = _format_list(p.get("confirming_evidence", []))
            df = _format_list(p.get("discriminating_features", []))
            queries = [f"{bg} {df}", cf, f"{p.get('reasoning', '')} diagnosis"]
        all_queries.append(queries[:5])
    print(f"  ✓ Phase 2 done: {len(all_queries)} query sets in {time.time() - t2:.1f}s")
    print(f"\n[Phase 3/4] Retrieving documents for {n} questions ...")
    t3 = time.time()
    try:
        from hyper.retrieval import create_retriever as _cr

        retriever = _cr(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
        if hasattr(retriever, "_lazy_init"):
            retriever._lazy_init()
    except Exception as e:
        raise RuntimeError(
            "Retrieval initialization failed; check dependencies and corpus preparation."
        ) from e
    all_docs = []
    for (i, queries) in enumerate(all_queries):
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        if retriever:
            k_per = max(1, total_docs // max(len(queries), 1))
            for q in queries:
                try:
                    (docs, scores) = retriever.retrieve(q, k=k_per)
                    if not docs or len(docs) != len(scores):
                        raise RuntimeError(
                            "Retriever returned empty or mismatched documents and scores."
                        )
                    for (doc, score) in zip(docs, scores):
                        doc_id = doc.get(
                            "id",
                            doc.get("title", str(hash(doc.get("content", "")[:100]))),
                        )
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = 0.0
                            doc_data[doc_id] = doc.copy()
                            doc_data[doc_id]["query_trace"] = []
                        try:
                            doc_scores[doc_id] += float(score)
                        except Exception:
                            doc_scores[doc_id] += 0.0
                        doc_data[doc_id]["query_trace"].append(q)
                except Exception as exc:
                    raise RuntimeError(
                        f"Retrieval failed for question {i + 1}."
                    ) from exc
        docs_sorted = []
        for doc_id in sorted(
            doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True
        ):
            doc = doc_data[doc_id]
            doc["fused_score"] = doc_scores[doc_id]
            docs_sorted.append(doc)
        all_docs.append(docs_sorted)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n} retrieved")
    print(f"  ✓ Phase 3 done: retrieval in {time.time() - t3:.1f}s")
    if retriever is not None:
        del retriever
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    print("  ✓ Retriever GPU memory freed")
    print(
        f"\n[Phase 4/4] Generating answers for {n} questions ({os.path.basename(generator_model)}) ..."
    )
    t4 = time.time()
    ans_messages = []
    for (i, qd) in enumerate(questions):
        options_text = "\n".join(
            [f"{k}. {v}" for (k, v) in sorted(qd["options"].items())]
        )
        docs = all_docs[i]
        ctx = (
            "\n\n".join(
                [
                    f"Document [{j + 1}] (Title: {d.get('title', 'Untitled')})\n{d.get('content', '')}"
                    for (j, d) in enumerate(docs[:total_docs])
                ]
            )
            or "No documents."
        )
        fmt_vars = {
            "context": ctx,
            "question": qd["question"],
            "options": options_text,
        }
        user = g_prompt["user"].format(**fmt_vars)
        ans_messages.append(
            [
                {"role": "system", "content": g_prompt["system"]},
                {"role": "user", "content": user},
            ]
        )
    ans_texts = _run_phase_in_subprocess(
        generator_model,
        ans_messages,
        gpu_mem,
        max_model_len,
        max_tokens,
        vllm_tensor_parallel_size,
    )
    print(f"  ✓ Phase 4 done: answers in {time.time() - t4:.1f}s")
    print(f"\n[Result] Compiling ...")
    results = []
    correct = 0
    total = 0
    for (i, qd) in enumerate(questions):
        correct_answer = qd.get("answer_idx", qd.get("answer", ""))
        raw_resp = ans_texts[i]
        predicted = parse_answer(raw_resp)
        is_correct = predicted.upper() == correct_answer.upper()
        if is_correct:
            correct += 1
        total += 1
        results.append(
            {
                "question_id": question_ids[i],
                "question": qd["question"],
                "options": qd["options"],
                "correct_answer": correct_answer,
                "modes": {
                    method: {
                        "num_queries": len(all_queries[i]),
                        "num_docs": len(all_docs[i]),
                        "queries": all_queries[i],
                        "plan": plans[i],
                        "retrieved_docs": _serialize_retrieved_docs_for_output(
                            all_docs[i]
                        ),
                        "raw_response": raw_resp,
                        "predicted_answer": predicted,
                        "is_correct": is_correct,
                    }
                },
            }
        )
    elapsed = time.time() - t_total
    accuracy = correct / total * 100 if total > 0 else 0
    summary = {
        "config": {
            "base_model": base_model,
            "question_dataset": question_dataset,
            "method": method,
            "total_evaluated": total,
            "total_docs": total_docs,
            "gpu_mem": gpu_mem,
            "vllm_tensor_parallel_size": vllm_tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
        },
        "timing": {
            "total_seconds": elapsed,
            "avg_per_question": elapsed / total if total else 0,
            "questions_per_minute": total / elapsed * 60 if elapsed > 0 else 0,
        },
        "planner_results": {
            "best_guess_accuracy": best_guess_accuracy,
            "best_guess_correct": best_guess_correct,
            "best_guess_total": total,
            "closest_alternative_accuracy": closest_alternative_accuracy,
            "closest_alternative_correct": closest_alternative_correct,
            "closest_alternative_rescue": closest_alternative_rescue,
            "top2_accuracy": top2_accuracy,
        },
        "mode_results": {
            method: {"correct": correct, "total": total, "accuracy": accuracy}
        },
        "timestamp": datetime.now().isoformat(),
    }
    print(f"\n{'=' * 70}")
    print(
        f"[{method}] {accuracy:.2f}% ({correct}/{total}) in {elapsed:.0f}s ({summary['timing']['questions_per_minute']:.1f} q/min)"
    )
    print(
        f"best_guess={best_guess_accuracy:.2f}% | closest_alt={closest_alternative_accuracy:.2f}% | top2={top2_accuracy:.2f}%"
    )
    print(f"{'=' * 70}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"{question_dataset}_{method}_{ts}.json")
    with open(out_file, "w") as f:
        json.dump(
            {"summary": summary, "results": results}, f, indent=2, ensure_ascii=False
        )
    print(f"✓ Saved: {out_file}")
    return {
        "method": method,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "file": out_file,
    }
