#!/usr/bin/env python3
"""
MedQA Dataset Loader for GRPO Training

Prepares the MedQA dataset for TRL GRPOTrainer by:
1. Loading MedQA questions from the MIRAGE benchmark
2. Pre-generating plans via a frozen planner (vLLM, temp=0)
3. Formatting prompts as (question + options + plan) for the rewriter
4. Outputting a HuggingFace Dataset with 'prompt' column (chat messages)

The plan pre-generation step is expensive, so results are cached to disk.
"""

import os
import sys
import json
import re
import hashlib
from typing import Dict, List, Any, Optional

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ============================================================================
# Rewriter prompt template (plan-conditioned, 3 queries)
# ============================================================================
REWRITER_GRPO_PROMPT = """Generate exactly 3 highly targeted, non-overlapping medical search queries to find evidence for this clinical question.

Question: {question}

Options:
{options}

Diagnostic Plan:
{plan_json}

RULES:
1. Output EXACTLY 3 queries, no more, no less.
2. Each query must target different information — no overlap.
3. Query 1: Find evidence supporting the best guess answer.
4. Query 2: Find comparative/distinguishing criteria between the best guess and the alternative.
5. Query 3: Find specific clinical, pathological, or mechanistic features.
6. Be concise and specific — each query should be a single search phrase.

Output format (strict):
Query 1: <query>
Query 2: <query>
Query 3: <query>"""


REWRITER_SYSTEM_PROMPT = "You are a medical search query expert. Generate precise, targeted search queries. Output ONLY the 3 queries in the exact format requested. No explanations."


# ============================================================================
# Plan generation prompt (same as PLANNING_V4_PROMPT in evaluate_medqa.py)
# ============================================================================
PLANNING_V4_PROMPT = """You are an expert medical diagnostician taking a medical licensing exam.

Question: {question}

Options:
{options}

Step 1: Identify the KEY DISCRIMINATING FEATURES that distinguish between the options.
Step 2: Make your BEST GUESS for the answer based on medical knowledge.
Step 3: Identify what SPECIFIC EVIDENCE would CONFIRM your answer.

Output in JSON:
{{
    "discriminating_features": ["2-3 features that distinguish between options"],
    "best_guess": "A/B/C/D",
    "reasoning": "brief explanation why this is the best answer",
    "confirming_evidence": ["1-3 specific facts that would confirm this answer"],
    "alternative_if_wrong": "A/B/C/D - only if uncertain"
}}"""

PLANNER_SYSTEM_PROMPT = "You are an expert medical diagnostician. Make your best diagnostic guess and identify what evidence would confirm it."


def load_medqa_benchmark(benchmark_path: Optional[str] = None, split: str = "medqa") -> Dict[str, Any]:
    """Load MedQA benchmark data.

    Args:
        benchmark_path: Path to benchmark.json.
        split: Dataset split key (default: "medqa").

    Returns:
        Dict mapping question_id -> question_data.
    """
    if benchmark_path is None:
        possible_paths = [
            os.path.join(_PROJECT_ROOT, "MIRAGE", "benchmark.json"),
            os.path.join(_PROJECT_ROOT, "..", "MIRAGE", "benchmark.json"),
            "/mnt/data1/home/hangeol/project/MIRAGE/benchmark.json",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                benchmark_path = path
                break
        if benchmark_path is None:
            raise FileNotFoundError("benchmark.json not found in any default location")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    dataset = benchmark[split]
    print(f"✓ Loaded {len(dataset)} {split} questions from {benchmark_path}")
    return dataset


def parse_plan_json(response: str) -> Dict[str, Any]:
    """Parse JSON plan from LLM response."""
    try:
        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def generate_plans_vllm(
    questions: List[Dict[str, Any]],
    question_ids: List[str],
    model_name: str,
    cache_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
) -> Dict[str, Dict[str, Any]]:
    """Generate plans for all questions using vLLM (temp=0, frozen planner).

    Args:
        questions: List of question dicts with 'question' and 'options'.
        question_ids: Corresponding question IDs.
        model_name: HuggingFace model name/path for the planner.
        cache_path: If set, load/save plans from/to this JSON file.
        tensor_parallel_size: vLLM tensor parallelism.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: Maximum model context length.

    Returns:
        Dict mapping question_id -> plan dict.
    """
    # Try loading from cache
    plans = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            plans = json.load(f)
        cached = sum(1 for qid in question_ids if qid in plans)
        print(f"✓ Loaded {cached}/{len(question_ids)} cached plans from {cache_path}")
        if cached == len(question_ids):
            return plans

    # Find questions that need plans
    missing_ids = [qid for qid in question_ids if qid not in plans]
    missing_questions = [q for q, qid in zip(questions, question_ids) if qid not in plans]

    if not missing_questions:
        return plans

    print(f"Generating plans for {len(missing_questions)} questions via vLLM ({model_name})...")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512, stop=["###", "\n\n\n"])

    # Build prompts
    prompts = []
    for q_data in missing_questions:
        question = q_data["question"]
        options = q_data["options"]
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        user_msg = PLANNING_V4_PROMPT.format(question=question, options=options_text)

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"system: {PLANNER_SYSTEM_PROMPT}\nuser: {user_msg}\nassistant:"

        prompts.append(prompt_text)

    # Generate
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for qid, output in zip(missing_ids, outputs):
        response_text = output.outputs[0].text if output.outputs else ""
        plan = parse_plan_json(response_text)
        # Ensure required fields
        plan.setdefault("discriminating_features", [])
        plan.setdefault("best_guess", "")
        plan.setdefault("reasoning", "")
        plan.setdefault("confirming_evidence", [])
        plan.setdefault("alternative_if_wrong", "")
        plans[qid] = plan

    # Clean up vLLM to free GPU memory for training
    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(plans, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(plans)} plans to {cache_path}")

    return plans


def format_rewriter_prompt(
    question: str,
    options: Dict[str, str],
    plan: Dict[str, Any],
) -> List[Dict[str, str]]:
    """Format a rewriter prompt as chat messages for TRL.

    Args:
        question: The medical question text.
        options: Dict of option labels to option text.
        plan: Plan dict from the frozen planner.

    Returns:
        List of chat message dicts (system + user).
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    plan_json = json.dumps(plan, indent=2, ensure_ascii=False)

    user_msg = REWRITER_GRPO_PROMPT.format(
        question=question,
        options=options_text,
        plan_json=plan_json,
    )

    return [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def build_grpo_dataset(
    benchmark_path: Optional[str] = None,
    split: str = "medqa",
    base_model: str = "google/gemma-2b-it",
    max_questions: Optional[int] = None,
    plan_cache_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
):
    """Build a HuggingFace Dataset for GRPO training.

    Each row contains:
        - prompt: chat messages (system + user) with plan-conditioned rewriter prompt
        - gold_answer: correct answer letter
        - question_id: question identifier
        - question: original question text
        - options: original options dict

    Args:
        benchmark_path: Path to benchmark.json.
        split: Dataset split.
        base_model: Model used for plan generation.
        max_questions: Limit number of questions (for dry runs).
        plan_cache_path: Cache file for pre-generated plans.
        tensor_parallel_size: vLLM tensor parallelism for plan generation.
        gpu_memory_utilization: vLLM GPU memory fraction.
        max_model_len: vLLM max model context length.

    Returns:
        datasets.Dataset
    """
    from datasets import Dataset

    raw_data = load_medqa_benchmark(benchmark_path, split)
    question_ids = sorted(raw_data.keys())

    if max_questions is not None:
        question_ids = question_ids[:max_questions]

    questions = [raw_data[qid] for qid in question_ids]

    # Default cache path
    if plan_cache_path is None:
        model_hash = hashlib.md5(base_model.encode()).hexdigest()[:8]
        plan_cache_path = os.path.join(
            _PROJECT_ROOT, "outputs", f"plan_cache_{split}_{model_hash}.json"
        )

    # Generate plans
    plans = generate_plans_vllm(
        questions=questions,
        question_ids=question_ids,
        model_name=base_model,
        cache_path=plan_cache_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Build dataset rows
    rows = []
    for qid, q_data in zip(question_ids, questions):
        plan = plans.get(qid, {})
        prompt = format_rewriter_prompt(q_data["question"], q_data["options"], plan)
        gold = q_data.get("answer_idx", q_data.get("answer", ""))

        rows.append({
            "prompt": prompt,
            "gold_answer": gold,
            "question_id": qid,
            "question": q_data["question"],
            "options": json.dumps(q_data["options"], ensure_ascii=False),
        })

    dataset = Dataset.from_list(rows)
    print(f"✓ Built GRPO dataset: {len(dataset)} examples")
    return dataset
