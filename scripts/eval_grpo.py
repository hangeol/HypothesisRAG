#!/usr/bin/env python3
"""
GRPO Integration Test: Compare Baseline vs Trained Rewriter

Runs MedQA evaluation on N questions comparing:
  (a) planning_v4 baseline (untrained rewriter)
  (b) planning_v4_grpo (trained rewriter with LoRA adapter)

Usage:
    python scripts/eval_grpo.py \
        --adapter_path outputs/rewriter_grpo_lora \
        --base_model google/gemma-2-9b-it \
        --n 50 \
        --retriever_name MedCPT \
        --corpus_name Textbooks
"""

import os
import sys
import json
import argparse
import re
from typing import Dict, List, Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare baseline vs GRPO-trained rewriter")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name/path")
    parser.add_argument("--benchmark_path", type=str, default=None, help="Path to benchmark.json")
    parser.add_argument("--split", type=str, default="medqa")
    parser.add_argument("-n", "--num_questions", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--retriever_name", type=str, default="MedCPT")
    parser.add_argument("--corpus_name", type=str, default="Textbooks")
    parser.add_argument("--total_docs", type=int, default=15)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--output_path", type=str, default=None, help="Save results JSON here")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_baseline_rewriter(
    question: str,
    options: Dict[str, str],
    plan: Dict[str, Any],
    vllm_generator,
    tokenizer,
    sampling_params,
) -> str:
    """Run baseline (untrained) rewriter via vLLM."""
    from data.medqa_loader import format_rewriter_prompt, REWRITER_SYSTEM_PROMPT

    messages = format_rewriter_prompt(question, options, plan)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = f"system: {messages[0]['content']}\nuser: {messages[1]['content']}\nassistant:"

    outputs = vllm_generator.generate([prompt_text], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""


def run_grpo_rewriter(
    question: str,
    options: Dict[str, str],
    plan: Dict[str, Any],
    model,
    tokenizer,
) -> str:
    """Run GRPO-trained rewriter (HF model with LoRA adapter)."""
    import torch
    from data.medqa_loader import format_rewriter_prompt

    messages = format_rewriter_prompt(question, options, plan)
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = f"system: {messages[0]['content']}\nuser: {messages[1]['content']}\nassistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
        )
    completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main():
    args = parse_args()

    # ====================================================================
    # 1. Load dataset + pre-generate plans
    # ====================================================================
    print("\n[1/5] Loading dataset and generating plans...")
    from data.medqa_loader import (
        load_medqa_benchmark,
        generate_plans_vllm,
    )

    raw_data = load_medqa_benchmark(args.benchmark_path, args.split)
    question_ids = sorted(raw_data.keys())[:args.num_questions]
    questions = [raw_data[qid] for qid in question_ids]

    plans = generate_plans_vllm(
        questions=questions,
        question_ids=question_ids,
        model_name=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # ====================================================================
    # 2. Initialize retriever + generator
    # ====================================================================
    print("\n[2/5] Initializing retriever and generator...")
    from retriever import create_retriever
    retriever = create_retriever(
        retriever_type="mirage",
        retriever_name=args.retriever_name,
        corpus_name=args.corpus_name,
    )

    from vllm import LLM, SamplingParams
    generator_llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    generator_tokenizer = generator_llm.get_tokenizer()
    gen_sampling = SamplingParams(temperature=0.0, max_tokens=512)
    rewrite_sampling = SamplingParams(temperature=0.0, max_tokens=256)

    # ====================================================================
    # 3. Load GRPO-trained rewriter
    # ====================================================================
    print("\n[3/5] Loading GRPO-trained rewriter adapter...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    rewriter_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if rewriter_tokenizer.pad_token is None:
        rewriter_tokenizer.pad_token = rewriter_tokenizer.eos_token

    rewriter_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    rewriter_model = PeftModel.from_pretrained(rewriter_base, args.adapter_path)
    rewriter_model.eval()
    print(f"✓ Loaded adapter from {args.adapter_path}")

    # ====================================================================
    # 4. Run evaluation
    # ====================================================================
    print(f"\n[4/5] Evaluating {len(question_ids)} questions...")
    from training.reward import (
        parse_queries_from_completion,
        parse_answer,
        format_generator_prompt,
        MIRAGE_SYSTEM_PROMPT,
    )

    results = {"baseline": [], "grpo": []}
    baseline_correct = 0
    grpo_correct = 0

    for idx, (qid, q_data) in enumerate(zip(question_ids, questions)):
        plan = plans.get(qid, {})
        gold = q_data.get("answer_idx", q_data.get("answer", ""))
        question = q_data["question"]
        options = q_data["options"]

        # --- Baseline Rewriter ---
        baseline_text = run_baseline_rewriter(
            question, options, plan, generator_llm, generator_tokenizer, rewrite_sampling
        )
        baseline_queries = parse_queries_from_completion(baseline_text)

        # --- GRPO Rewriter ---
        grpo_text = run_grpo_rewriter(
            question, options, plan, rewriter_model, rewriter_tokenizer
        )
        grpo_queries = parse_queries_from_completion(grpo_text)

        # --- Retrieve + Generate for both ---
        for mode, queries in [("baseline", baseline_queries), ("grpo", grpo_queries)]:
            # Retrieve
            all_docs = {}
            k_per = max(1, args.total_docs // len(queries)) if queries else args.total_docs
            for q in queries:
                try:
                    docs, scores = retriever.retrieve(q, k=k_per)
                    for d, s in zip(docs, scores):
                        did = d.get("id", d.get("title", ""))
                        if did not in all_docs:
                            all_docs[did] = (d, s)
                        else:
                            all_docs[did] = (d, max(all_docs[did][1], s))
                except Exception:
                    pass

            retrieved = [d for d, _ in sorted(all_docs.values(), key=lambda x: x[1], reverse=True)]

            # Generate
            user_prompt = format_generator_prompt(question, options, retrieved)
            messages = [
                {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            try:
                prompt_text = generator_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt_text = f"system: {MIRAGE_SYSTEM_PROMPT}\nuser: {user_prompt}\nassistant:"

            gen_out = generator_llm.generate([prompt_text], gen_sampling, use_tqdm=False)
            raw_answer = gen_out[0].outputs[0].text if gen_out and gen_out[0].outputs else ""
            predicted = parse_answer(raw_answer)
            correct = predicted.upper() == gold.upper()

            if mode == "baseline":
                baseline_correct += int(correct)
            else:
                grpo_correct += int(correct)

            results[mode].append({
                "question_id": qid,
                "gold": gold,
                "predicted": predicted,
                "correct": correct,
                "queries": queries,
                "num_docs": len(retrieved),
            })

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(question_ids)}] "
                  f"Baseline: {baseline_correct}/{idx+1} ({100*baseline_correct/(idx+1):.1f}%) | "
                  f"GRPO: {grpo_correct}/{idx+1} ({100*grpo_correct/(idx+1):.1f}%)")

    # ====================================================================
    # 5. Summary
    # ====================================================================
    n = len(question_ids)
    print(f"\n{'='*60}")
    print(f"Results on {n} MedQA questions:")
    print(f"  Baseline (untrained):  {baseline_correct}/{n} ({100*baseline_correct/n:.1f}%)")
    print(f"  GRPO (trained):        {grpo_correct}/{n} ({100*grpo_correct/n:.1f}%)")
    diff = grpo_correct - baseline_correct
    sign = "+" if diff >= 0 else ""
    print(f"  Difference:            {sign}{diff} ({sign}{100*diff/n:.1f}%)")
    print(f"{'='*60}")

    # Save results
    output_path = args.output_path or os.path.join(args.adapter_path, "eval_results.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    summary = {
        "num_questions": n,
        "baseline_accuracy": baseline_correct / n,
        "grpo_accuracy": grpo_correct / n,
        "baseline_correct": baseline_correct,
        "grpo_correct": grpo_correct,
        "args": vars(args),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
