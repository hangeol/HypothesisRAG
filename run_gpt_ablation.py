#!/usr/bin/env python3
"""
GPT-4o / GPT-4o-mini ablation for prompt comparison.
Runs hv5-rv5-gv1 and hv7-rv10-gv1 on OpenAI models.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts import HYPOTHESIS_PROMPTS, REWRITING_PROMPTS, GENERATOR_PROMPTS


# ============================================================================
# Defaults
# ============================================================================
DEFAULT_SPLIT = "medqa"
DEFAULT_MAX_QUESTIONS = 1273
DEFAULT_CONCURRENCY = 120
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o"]
DEFAULT_COMBOS = ["v5-v5", "v7-v10"]
DEFAULT_GENERATOR_PROMPT = "v1"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_RETRIEVER = "MedCPT"
DEFAULT_CORPUS = "Textbooks"
MAX_RETRIES = 3


# ============================================================================
# CLI / Config helpers
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT ablation with selected prompt combinations.")
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--max-questions", "-n", type=int, default=DEFAULT_MAX_QUESTIONS)
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--combos", nargs="+", default=DEFAULT_COMBOS,
                        help="Prompt combos in hv-rv format, e.g. v5-v5 v7-v10")
    parser.add_argument("--generator-prompt", type=str, default=DEFAULT_GENERATOR_PROMPT, choices=["v1", "v2"])
    parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--retriever", type=str, default=DEFAULT_RETRIEVER)
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS)
    return parser.parse_args()


def resolve_benchmark_path(explicit_path: Optional[str]) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        explicit_path,
        os.path.join(script_dir, "MIRAGE", "benchmark.json"),
        os.path.join(script_dir, "data", "benchmark.json"),
        os.path.join(script_dir, "..", "MIRAGE", "benchmark.json"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("benchmark.json not found. Use --benchmark-path to specify it.")


def maybe_load_api_key_from_bashrc() -> Optional[str]:
    bashrc = os.path.expanduser("~/.bashrc")
    if not os.path.exists(bashrc):
        return None
    try:
        text = open(bashrc, "r", encoding="utf-8").read()
    except Exception:
        return None
    m = re.search(r'export\s+OPENAI_API_KEY\s*=\s*"([^"]+)"', text)
    if not m:
        m = re.search(r"export\s+OPENAI_API_KEY\s*=\s*'([^']+)'", text)
    if not m:
        m = re.search(r"export\s+OPENAI_API_KEY\s*=\s*([^\s#]+)", text)
    return m.group(1).strip() if m else None


def parse_combos(combo_tokens: List[str]) -> List[Tuple[str, str]]:
    combos: List[Tuple[str, str]] = []
    for token in combo_tokens:
        if "-" not in token:
            raise ValueError(f"Invalid combo '{token}'. Expected hv-rv format (e.g. v5-v5).")
        hv, rv = token.split("-", 1)
        if hv not in HYPOTHESIS_PROMPTS:
            raise ValueError(f"Unknown hypothesis prompt: {hv}")
        if rv not in REWRITING_PROMPTS:
            raise ValueError(f"Unknown rewriting prompt: {rv}")
        combos.append((hv, rv))
    return combos


# ============================================================================
# Data loading
# ============================================================================
def load_benchmark(benchmark_path: str, split: str) -> Dict[str, Dict[str, Any]]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get(split, data)
    if isinstance(raw, list):
        return {str(i): q for i, q in enumerate(raw)}
    return raw


# ============================================================================
# Async OpenAI calls
# ============================================================================
async def call_openai(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    timeout = aiohttp.ClientTimeout(total=120)

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    if resp.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        if attempt == MAX_RETRIES - 1:
                            err = await resp.text()
                            return f"Error: {resp.status} {err[:200]}"
                        await asyncio.sleep(1)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return f"Error: {e}"
                await asyncio.sleep(1)
    return "Error: max retries"


async def batch_call_ordered(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    api_key: str,
    model: str,
    messages_list: List[List[Dict[str, str]]],
    temperature: float,
    desc: str,
) -> List[str]:
    tasks = [
        asyncio.create_task(call_openai(session, sem, api_key, model, msg, temperature))
        for msg in messages_list
    ]
    total = len(tasks)
    done = 0
    for fut in asyncio.as_completed(tasks):
        await fut
        done += 1
        if done % 100 == 0 or done == total:
            print(f"  [{desc}] {done}/{total}")
    return await asyncio.gather(*tasks)


# ============================================================================
# Parsing helpers
# ============================================================================
def parse_json_from_text(text: str) -> Dict[str, Any]:
    if not text or text.startswith("Error"):
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    # Handle JSON wrapped in extra explanation text.
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {}


def parse_queries(text: str) -> List[str]:
    if not text or text.startswith("Error"):
        return []
    queries: List[str] = []
    for line in text.strip().split("\n"):
        m = re.match(r"\s*Query\s*\d+\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            query = m.group(1).strip()
            if query:
                queries.append(query)
    if not queries:
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith("{"):
                queries.append(line)
    return queries[:3]


def parse_answer(text: str) -> str:
    if not text or text.startswith("Error"):
        return ""

    data = parse_json_from_text(text)
    answer = data.get("answer_choice", data.get("answer", ""))
    if answer:
        return answer.strip().upper()[:1]

    m = re.search(r'"answer_choice"\s*:\s*"([A-Z])"', text)
    if m:
        return m.group(1)

    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


def _format_list(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value) if value else ""


# ============================================================================
# Main evaluation pipeline
# ============================================================================
async def run_evaluation(
    model_name: str,
    hv: str,
    rv: str,
    api_key: str,
    qids: List[str],
    questions: List[Dict[str, Any]],
    concurrency: int,
    generator_prompt: str,
    output_dir: str,
    retriever_name: str,
    corpus_name: str,
) -> Tuple[str, float]:
    tag = f"{model_name}_hv{hv}_rv{rv}_gv{generator_prompt}"
    print(f"\n{'='*70}")
    print(f"Starting: {tag}")
    print(f"Questions: {len(questions)} | Concurrency: {concurrency}")
    print(f"{'='*70}")

    sem = asyncio.Semaphore(concurrency)
    h_prompt = HYPOTHESIS_PROMPTS[hv]
    r_prompt = REWRITING_PROMPTS[rv]
    g_prompt = GENERATOR_PROMPTS[generator_prompt]

    async with aiohttp.ClientSession() as session:
        t0 = time.time()

        # Phase 1: Hypothesis generation
        print(f"  Phase 1: Hypothesis ({hv})")
        h_messages: List[List[Dict[str, str]]] = []
        for qd in questions:
            options_text = "\n".join([f"{k}. {v}" for k, v in sorted(qd["options"].items())])
            user = h_prompt["user"].format(question=qd["question"], options=options_text)
            h_messages.append([
                {"role": "system", "content": h_prompt["system"]},
                {"role": "user", "content": user},
            ])

        h_texts = await batch_call_ordered(
            session=session,
            sem=sem,
            api_key=api_key,
            model=model_name,
            messages_list=h_messages,
            temperature=0,
            desc="hypothesis",
        )
        plans = [parse_json_from_text(t) for t in h_texts]
        print(f"  Phase 1 done: {sum(1 for p in plans if p)}/{len(plans)} parsed")

        # Phase 2: Rewriting
        print(f"  Phase 2: Rewriting ({rv})")
        r_messages: List[List[Dict[str, str]]] = []
        for i, qd in enumerate(questions):
            plan = plans[i]
            options_text = "\n".join([f"{k}. {v}" for k, v in sorted(qd["options"].items())])

            bg_letter = (plan.get("best_guess", "") or "").strip().upper().rstrip(".")
            bg_text = plan.get("best_guess_text") or (
                f"{bg_letter}. {qd['options'].get(bg_letter, bg_letter)}"
                if bg_letter in qd.get("options", {})
                else bg_letter
            )
            alt_letter = (plan.get("alternative_if_wrong", "") or "").strip().upper().rstrip(".")
            alt_text = (
                f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
                if alt_letter in qd.get("options", {})
                else alt_letter
            )

            user = r_prompt["user"].format(
                question=qd["question"],
                options=options_text,
                best_guess=plan.get("best_guess", ""),
                best_guess_text=bg_text,
                reasoning=plan.get("reasoning", ""),
                confirming_evidence=_format_list(plan.get("confirming_evidence", [])),
                discriminating_features=_format_list(plan.get("discriminating_features", [])),
                alternative_if_wrong=plan.get("alternative_if_wrong", ""),
                alternative_text=alt_text,
            )
            r_messages.append([
                {"role": "system", "content": r_prompt["system"]},
                {"role": "user", "content": user},
            ])

        r_texts = await batch_call_ordered(
            session=session,
            sem=sem,
            api_key=api_key,
            model=model_name,
            messages_list=r_messages,
            temperature=0,
            desc="rewriter",
        )
        all_queries = [parse_queries(t) for t in r_texts]
        print(f"  Phase 2 done: {sum(1 for q in all_queries if q)}/{len(all_queries)} have queries")

        # Phase 3: Retrieval
        print(f"  Phase 3: Retrieval ({retriever_name}/{corpus_name})")
        from retriever import create_retriever

        retriever = create_retriever(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )

        all_docs: List[List[Dict[str, Any]]] = []
        for i, queries in enumerate(all_queries):
            doc_scores: Dict[str, float] = {}
            doc_data: Dict[str, Dict[str, Any]] = {}
            k_per_query = max(1, 15 // max(len(queries), 1))

            for q in queries:
                try:
                    docs, scores = retriever.retrieve(q, k=k_per_query)
                    for doc, score in zip(docs, scores):
                        doc_id = doc.get("id", doc.get("title", str(hash(doc.get("content", "")[:100]))))
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = 0.0
                            doc_data[doc_id] = doc.copy()
                        doc_scores[doc_id] += float(score)
                except Exception:
                    continue

            ranked_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
            all_docs.append([doc_data[doc_id] for doc_id in ranked_ids[:25]])

            if (i + 1) % 200 == 0:
                print(f"    Retrieved {i+1}/{len(all_queries)}")

        print("  Phase 3 done")

        # Phase 4: Final answer generation
        print(f"  Phase 4: Generator ({generator_prompt})")
        g_messages: List[List[Dict[str, str]]] = []
        for i, qd in enumerate(questions):
            docs = all_docs[i]
            options_text = "\n".join([f"{k}. {v}" for k, v in sorted(qd["options"].items())])

            ctx_parts = []
            for j, doc in enumerate(docs[:25]):
                ctx_parts.append(
                    f"Document [{j+1}] (Title: {doc.get('title', 'Untitled')})\n{doc.get('content', '')}"
                )
            context = "\n\n".join(ctx_parts) if ctx_parts else "No documents."

            fmt_vars = {
                "context": context,
                "question": qd["question"],
                "options": options_text,
                "hypothesis_summary": "",
                "queries_summary": "",
            }
            if generator_prompt == "v2":
                plan = plans[i]
                fmt_vars["hypothesis_summary"] = (
                    f"Best guess: {plan.get('best_guess', '')} - {plan.get('reasoning', '')}"
                )
                fmt_vars["queries_summary"] = "\n".join(
                    f"  {idx+1}. {q}" for idx, q in enumerate(all_queries[i])
                )

            user = g_prompt["user"].format(**fmt_vars)

            # Light input size guard for API stability
            while len(user) > 24000 and ctx_parts:
                ctx_parts.pop()
                fmt_vars["context"] = "\n\n".join(ctx_parts) if ctx_parts else "No documents."
                user = g_prompt["user"].format(**fmt_vars)

            g_messages.append([
                {"role": "system", "content": g_prompt["system"]},
                {"role": "user", "content": user},
            ])

        g_texts = await batch_call_ordered(
            session=session,
            sem=sem,
            api_key=api_key,
            model=model_name,
            messages_list=g_messages,
            temperature=0,
            desc="generator",
        )

        # Scoring
        correct = 0
        results = []
        for i, (qid, qd) in enumerate(zip(qids, questions)):
            predicted = parse_answer(g_texts[i])
            gold = qd.get("answer_idx", qd.get("answer", ""))
            is_correct = bool(predicted and gold and predicted.upper() == gold.upper())
            if is_correct:
                correct += 1

            results.append({
                "question_id": qid,
                "predicted": predicted,
                "gold": gold,
                "correct": is_correct,
            })

        elapsed = time.time() - t0
        accuracy = (correct / len(questions) * 100) if questions else 0.0

        print(f"\n  [{tag}] {accuracy:.2f}% ({correct}/{len(questions)}) in {elapsed:.0f}s")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(
            output_dir,
            f"medqa_{model_name.replace('/', '_')}_hv{hv}_rv{rv}_gv{generator_prompt}_{timestamp}.json",
        )
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "model": model_name,
                        "hypothesis_prompt": hv,
                        "rewriting_prompt": rv,
                        "generator_prompt": generator_prompt,
                        "accuracy_pct": accuracy,
                        "correct": correct,
                        "total": len(questions),
                        "elapsed_seconds": elapsed,
                        "concurrency": concurrency,
                    },
                    "results": results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  Saved: {outfile}")

        return tag, accuracy


async def main() -> None:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = maybe_load_api_key_from_bashrc()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        print("Error: OPENAI_API_KEY not set. Export it or add it to ~/.bashrc")
        sys.exit(1)

    benchmark_path = resolve_benchmark_path(args.benchmark_path)
    data = load_benchmark(benchmark_path, args.split)
    qids = sorted(data.keys())[:args.max_questions]
    questions = [data[qid] for qid in qids]
    combos = parse_combos(args.combos)

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Benchmark: {benchmark_path}")
    print(f"Split: {args.split}")
    print(f"Questions: {len(questions)}")
    print(f"Models: {args.models}")
    print(f"Combos: {combos}")
    print(f"Generator: {args.generator_prompt}")
    print(f"Concurrency: {args.concurrency}")

    all_results: List[Tuple[str, float]] = []

    for model in args.models:
        for hv, rv in combos:
            tag, acc = await run_evaluation(
                model_name=model,
                hv=hv,
                rv=rv,
                api_key=api_key,
                qids=qids,
                questions=questions,
                concurrency=args.concurrency,
                generator_prompt=args.generator_prompt,
                output_dir=args.output_dir,
                retriever_name=args.retriever,
                corpus_name=args.corpus,
            )
            all_results.append((tag, acc))

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    for tag, acc in sorted(all_results, key=lambda x: -x[1]):
        print(f"  {acc:6.2f}%  {tag}")


if __name__ == "__main__":
    asyncio.run(main())
