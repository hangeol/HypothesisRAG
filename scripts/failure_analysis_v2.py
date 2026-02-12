#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Failure Analysis v2: Direct vs CoT RAG (Evidence-audit + Rule-based labeling)

Key design:
- LLM does ONLY "evidence audit" (facts): which option is supported by snippets? noise? ambiguity?
- Final label is deterministic rule-based -> avoids label collapse (e.g., all F3).

Primary labels:
- R-GAP, R-NOISE, R-TRAP, U-FAIL, I-FAIL, AMBIG, ERROR

SET1 additionally records rescue_mechanism:
- COT_RETRIEVAL_REPAIR, COT_REASONING_REPAIR, PARAMETRIC_KNOWLEDGE_RESCUE, OTHER

OpenAI API:
- Uses Responses API: POST https://api.openai.com/v1/responses
  docs: instructions, max_output_tokens, output_text. (See OpenAI API reference)
"""

import argparse
import asyncio
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


# -------------------------
# IO helpers
# -------------------------

def to_str(x: Any) -> str:
    return "" if x is None else str(x)

def load_json_file(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {file_path}")

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {file_path}: {e}")

    if isinstance(data, dict) and "results" in data:
        rows = data["results"]
    elif isinstance(data, list):
        rows = data
    else:
        rows = [data]

    out: Dict[str, Any] = {}
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        qid = item.get("question_id") or item.get("id") or item.get("qid") or i
        qid = to_str(qid).strip()
        if qid:
            out[qid] = item
    return out

def normalize_options(raw_opts: Any) -> Dict[str, str]:
    """
    Accepts:
    - dict: {"A": "...", "B": "..."} or {"0": "..."}
    - list: ["...", "..."] -> map to A,B,C,...
    """
    if raw_opts is None:
        return {}
    if isinstance(raw_opts, dict):
        # ensure str keys
        return {to_str(k).strip(): to_str(v) for k, v in raw_opts.items()}
    if isinstance(raw_opts, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        out = {}
        for i, v in enumerate(raw_opts):
            k = letters[i] if i < len(letters) else str(i)
            out[k] = to_str(v)
        return out
    # fallback
    return {}

def get_mode_block(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    modes = item.get("modes", {})
    if isinstance(modes, dict) and mode in modes and isinstance(modes[mode], dict):
        return modes[mode]
    # Backward compatibility: some result files are flat (no modes block)
    flat_keys = {
        "predicted_answer",
        "prediction",
        "answer_choice",
        "pred",
        "is_correct",
        "raw_response",
        "raw_answer",
        "retrieved_docs",
        "retrieved_passages",
        "snippets",
    }
    if isinstance(item, dict) and any(k in item for k in flat_keys):
        return item
    return {}

def get_correct_answer(item: Dict[str, Any]) -> str:
    for k in ["correct_answer", "gold_answer", "gold", "answer"]:
        if k in item and item[k] is not None:
            return to_str(item.get(k)).strip()
    return ""

def get_predicted_answer(item: Dict[str, Any], mode: str) -> Optional[str]:
    m = get_mode_block(item, mode)
    for k in ["predicted_answer", "prediction", "answer_choice", "pred"]:
        if k in m and m[k] is not None:
            return to_str(m[k]).strip()

    for raw_key in ["raw_response", "raw_answer", "response"]:
        raw = m.get(raw_key)
        if not isinstance(raw, str):
            continue
        obj = extract_json_from_text(raw)
        if isinstance(obj, dict):
            for k in ["answer_choice", "predicted_answer", "prediction", "pred"]:
                if k in obj and obj[k] is not None:
                    return to_str(obj[k]).strip()
    return None

def get_correctness(item: Dict[str, Any], mode: str) -> Optional[bool]:
    m = get_mode_block(item, mode)
    if "is_correct" in m and isinstance(m["is_correct"], bool):
        return m["is_correct"]

    pred = get_predicted_answer(item, mode)
    gold = get_correct_answer(item)
    if pred is None or not gold:
        return None
    return pred.strip() == gold.strip()

def truncate_text(s: str, max_chars: Optional[int]) -> str:
    if max_chars is None:
        return s
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_chars else (s[:max_chars] + "...")

def get_retrieved_docs(item: Dict[str, Any], mode: str, top_n: int, max_chars: Optional[int]) -> List[Dict[str, Any]]:
    m = get_mode_block(item, mode)
    docs = (
        m.get("retrieved_docs")
        or m.get("retrieved_passages")
        or m.get("snippets")
        or []
    )
    docs = docs[:top_n] if top_n > 0 else docs

    out = []
    for i, d in enumerate(docs, 1):
        dd = dict(d) if isinstance(d, dict) else {"content": to_str(d)}
        doc_id = dd.get("id") or dd.get("doc_id") or f"{mode}_doc_{i}"
        title = dd.get("title") or ""
        content = dd.get("content") or dd.get("text") or dd.get("chunk") or ""
        out.append({
            "id": to_str(doc_id),
            "title": to_str(title),
            "content": truncate_text(to_str(content), max_chars),
        })
    return out


# -------------------------
# OpenAI Responses API
# -------------------------

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()

    # Strip markdown fences if any
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            t = "\n".join(lines[1:-1]).strip()

    # Try direct parse
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Find first balanced {...}
    brace = 0
    start = -1
    for i, ch in enumerate(t):
        if ch == "{":
            if brace == 0:
                start = i
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0 and start >= 0:
                chunk = t[start:i+1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    continue

    # Regex fallback
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None

async def openai_responses_json(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    model: str,
    instructions: str,
    user_input: str,
    max_output_tokens: int = 1200,
    temperature: float = 0.0,
    timeout_s: int = 120,
    max_retries: int = 6,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Calls POST /v1/responses, expects model to output JSON-only text.
    Parses response.output_text into dict.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "text": {"format": {"type": "text"}},
    }

    def get_output_text(data: Dict[str, Any]) -> str:
        """
        Responses API compatibility:
        - Some SDK/versions expose `output_text` at top-level.
        - Others return text under output[].content[].text.
        """
        out_text = to_str(data.get("output_text", "")).strip()
        if out_text:
            return out_text

        chunks: List[str] = []
        output = data.get("output", [])
        if isinstance(output, list):
            for block in output:
                if not isinstance(block, dict):
                    continue
                content = block.get("content", [])
                if not isinstance(content, list):
                    continue
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    text = c.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
        return "\n".join(chunks).strip()

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as resp:
                    status = resp.status
                    body_txt = await resp.text()

                    if status == 200:
                        try:
                            data = json.loads(body_txt)
                        except json.JSONDecodeError as e:
                            if debug:
                                print(f"[openai] Response body JSON parse failed: {e}, preview: {body_txt[:400]}")
                            return {"_error": "response_json_parse_failed", "_raw": body_txt[:2000]}
                        out_text = get_output_text(data)
                        if not out_text or not out_text.strip():
                            if debug:
                                print(f"[openai] Empty output_text. Response keys: {list(data.keys())}")
                            return {"_error": "empty_output_text", "_raw": json.dumps(data, ensure_ascii=False)[:2000]}
                        obj = extract_json_from_text(out_text)
                        if obj is None:
                            if debug:
                                print("[openai] JSON parse failed. preview:", out_text[:400])
                            return {"_error": "json_parse_failed", "_raw": out_text[:2000]}
                        return obj

                    if debug:
                        print(f"[openai] status={status} attempt={attempt+1} body={body_txt[:300]}")

                    # auth errors: do not retry
                    if status in (401, 403):
                        return {"_error": f"http_{status}", "_raw": body_txt[:1000]}

                    # retryable
                    if status in (429, 500, 502, 503, 504):
                        await asyncio.sleep(min(2 ** attempt, 30) + random.random())
                        continue

                    return {"_error": f"http_{status}", "_raw": body_txt[:1000]}

            except asyncio.TimeoutError:
                if debug:
                    print(f"[openai] timeout attempt={attempt+1}")
                await asyncio.sleep(min(2 ** attempt, 30) + random.random())
            except Exception as e:
                if debug:
                    print(f"[openai] exception attempt={attempt+1}: {e}")
                await asyncio.sleep(min(2 ** attempt, 30) + random.random())

    return {"_error": "failed_after_retries"}


async def probe_api(session, semaphore, api_key, model, debug=False):
    instr = "Return JSON only."
    user = 'Return exactly: {"ok": true}'
    res = await openai_responses_json(
        session=session,
        semaphore=semaphore,
        api_key=api_key,
        model=model,
        instructions=instr,
        user_input=user,
        max_output_tokens=50,
        temperature=0.0,
        timeout_s=30,
        debug=debug,
    )
    if res.get("ok") is not True:
        raise RuntimeError(f"API probe failed for model={model}: {res}")


# -------------------------
# Evidence audit prompt
# -------------------------

def format_docs_for_prompt(docs: List[Dict[str, Any]], tag: str) -> str:
    if not docs:
        return f"{tag} Retrieved Docs: None"
    lines = [f"{tag} Retrieved Docs ({len(docs)}):"]
    for i, d in enumerate(docs, 1):
        doc_id = d.get("id", f"{tag.lower()}_{i}")
        title = d.get("title", "")
        content = d.get("content", "")
        lines.append(f"\n[{tag} Doc {i} | id={doc_id}]")
        if title:
            lines.append(f"Title: {title}")
        lines.append(f"Content: {content}")
    return "\n".join(lines)

async def evidence_audit(
    question: str,
    options: Dict[str, str],
    gold: str,
    direct_pred: str,
    cot_pred: str,
    direct_docs: List[Dict[str, Any]],
    cot_docs: List[Dict[str, Any]],
    model: str,
    api_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    debug: bool,
) -> Dict[str, Any]:
    opts_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    gold_text = options.get(gold, "")
    direct_text = options.get(direct_pred, "")
    cot_text = options.get(cot_pred, "")

    user = f"""
You are doing an EVIDENCE AUDIT for a medical multiple-choice question.

Goal: extract minimal factual signals from retrieved documents. Do NOT diagnose "failure cause" directly.

Question:
{question}

Options:
{opts_str}

Gold: {gold} | {gold_text}
Direct_pred: {direct_pred} | {direct_text}
CoT_pred: {cot_pred} | {cot_text}

{format_docs_for_prompt(direct_docs, "Direct")}
{format_docs_for_prompt(cot_docs, "CoT")}

Instructions:
For EACH of (Direct docs) and (CoT docs), do:
1) Identify up to 3 snippets that SUPPORT the GOLD option (if any).
2) Identify up to 3 snippets that SUPPORT the model's PREDICTED option (Direct_pred for Direct, CoT_pred for CoT) (if any).
3) Mark doc_ids that are mostly irrelevant to deciding among the options.

Snippets must be <= 20 words and copied from the doc (verbatim).
If no supporting snippet exists, return empty list.
Also mark whether the gold-support you found is DECISIVE for choosing gold over distractors.

Finally, judge whether the QUESTION/OPTIONS are inherently ambiguous:
- "is_ambiguous": true only if 2+ options remain plausible even with typical textbook knowledge.
- Do NOT mark ambiguous just because retrieved docs are missing/irrelevant or evidence is not found.

Output JSON ONLY:
{{
  "direct": {{
    "total_docs": {len(direct_docs)},
    "irrelevant_doc_ids": ["..."],
    "gold_support": [{{"doc_id":"...","quote":"<=20 words","strength":"strong|weak"}}],
    "pred_support": [{{"doc_id":"...","quote":"<=20 words","strength":"strong|weak"}}],
    "gold_decisive": true/false,
    "pred_decisive": true/false
  }},
  "cot": {{
    "total_docs": {len(cot_docs)},
    "irrelevant_doc_ids": ["..."],
    "gold_support": [{{"doc_id":"...","quote":"<=20 words","strength":"strong|weak"}}],
    "pred_support": [{{"doc_id":"...","quote":"<=20 words","strength":"strong|weak"}}],
    "gold_decisive": true/false,
    "pred_decisive": true/false
  }},
  "ambiguity": {{
    "is_ambiguous": true/false,
    "reason": "<=20 words",
    "plausible_options": ["A","B"]
  }}
}}
"""

    instr = "You are a careful medical evidence auditor. Output ONLY valid JSON text."
    obj = await openai_responses_json(
        session=session,
        semaphore=semaphore,
        api_key=api_key,
        model=model,
        instructions=instr,
        user_input=user,
        max_output_tokens=1400,
        temperature=0.0,
        timeout_s=150,
        debug=debug,
    )
    return obj


# -------------------------
# Rule-based labeling
# -------------------------

@dataclass
class MethodSignals:
    total_docs: int
    irrelevant: int
    noise_ratio: float
    gold_support_n: int
    pred_support_n: int
    gold_decisive: Optional[bool]
    pred_decisive: Optional[bool]

def method_signals(audit_block: Dict[str, Any]) -> MethodSignals:
    total = int(audit_block.get("total_docs", 0) or 0)
    irr_ids = audit_block.get("irrelevant_doc_ids", []) or []
    irr = len(irr_ids)
    noise_ratio = (irr / total) if total > 0 else 1.0
    gold_support_n = len(audit_block.get("gold_support", []) or [])
    pred_support_n = len(audit_block.get("pred_support", []) or [])
    gold_decisive = audit_block.get("gold_decisive", None)
    pred_decisive = audit_block.get("pred_decisive", None)
    return MethodSignals(total, irr, noise_ratio, gold_support_n, pred_support_n, gold_decisive, pred_decisive)

def label_wrong_prediction(
    sig: MethodSignals,
    ambig: bool,
    noise_thr: float = 0.60,
) -> str:
    """
    Label a method that answered WRONG, based on evidence signals.
    """
    gold_present = sig.gold_support_n > 0
    pred_present = sig.pred_support_n > 0

    # Ambiguity should be conservative:
    # require competing evidence for both gold and predicted options, but neither decisive.
    if (
        ambig
        and gold_present
        and pred_present
        and sig.gold_decisive is False
        and sig.pred_decisive is False
    ):
        return "AMBIG"

    if not gold_present:
        # retrieval-side failure
        if pred_present:
            return "R-TRAP"
        if sig.noise_ratio >= noise_thr:
            return "R-NOISE"
        return "R-GAP"

    # gold evidence exists but still wrong => utilization/inference side
    if sig.gold_decisive is True:
        return "I-FAIL"
    return "U-FAIL"

def combine_set2_label(
    direct_label: str,
    cot_label: str,
) -> str:
    """
    Produce a single primary label for SET2 (both wrong).
    Prefer retrieval failures if either indicates retrieval gap/noise/trap;
    otherwise utilization/inference; ambiguity; error.
    """
    if "ERROR" in (direct_label, cot_label):
        return "ERROR"
    # Avoid AMBIG inflation when only one side is marked ambiguous.
    if direct_label == "AMBIG" and cot_label == "AMBIG":
        return "AMBIG"

    retrieval = {"R-GAP", "R-NOISE", "R-TRAP"}
    if direct_label in retrieval and cot_label in retrieval:
        # if either is TRAP, treat as TRAP; else if either NOISE; else GAP
        if "R-TRAP" in (direct_label, cot_label):
            return "R-TRAP"
        if "R-NOISE" in (direct_label, cot_label):
            return "R-NOISE"
        return "R-GAP"

    if direct_label in retrieval or cot_label in retrieval:
        # mixed: still primarily retrieval issue
        if "R-TRAP" in (direct_label, cot_label):
            return "R-TRAP"
        if "R-NOISE" in (direct_label, cot_label):
            return "R-NOISE"
        return "R-GAP"

    # both are utilization/inference
    if "I-FAIL" in (direct_label, cot_label):
        return "I-FAIL"
    return "U-FAIL"

def rescue_mechanism(set1_direct_label: str, direct_sig: MethodSignals, cot_sig: MethodSignals, cot_correct_without_gold: bool) -> str:
    """
    For SET1 (direct wrong, cot correct) explain mechanism.
    """
    direct_gold = direct_sig.gold_support_n > 0
    cot_gold = cot_sig.gold_support_n > 0

    if (not direct_gold) and cot_gold:
        return "COT_RETRIEVAL_REPAIR"
    if direct_gold and cot_gold:
        return "COT_REASONING_REPAIR"
    if (not cot_gold) and cot_correct_without_gold:
        return "PARAMETRIC_KNOWLEDGE_RESCUE"
    return "OTHER"


# -------------------------
# Caching / output
# -------------------------

def load_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            j = json.loads(line)
            qid = to_str(j.get("question_id")).strip()
            if qid:
                out[qid] = j
        except Exception:
            continue
    return out

def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


# -------------------------
# Report
# -------------------------

PRIMARY_LABELS = ["R-GAP", "R-NOISE", "R-TRAP", "U-FAIL", "I-FAIL", "AMBIG", "ERROR"]

LABEL_PRETTY = {
    "R-GAP": "Retrieval Gap (gold evidence absent)",
    "R-NOISE": "Retrieval Noise (mostly irrelevant)",
    "R-TRAP": "Retrieval Trap (supports wrong option)",
    "U-FAIL": "Evidence Underuse (gold present, not used)",
    "I-FAIL": "Inference Failure (gold decisive, still wrong)",
    "AMBIG": "Ambiguity / underspecified",
    "ERROR": "LLM/API error",
}

def count_labels(records: List[Dict[str, Any]]) -> Dict[str, int]:
    c = defaultdict(int)
    for r in records:
        lab = r.get("primary", "ERROR")
        c[lab] += 1
    return dict(c)

def markdown_table(counts: Dict[str, int], total: int) -> str:
    lines = ["| Category | Count | Percentage |", "|---|---:|---:|"]
    for lab in PRIMARY_LABELS:
        n = counts.get(lab, 0)
        pct = (n / total * 100) if total else 0.0
        lines.append(f"| {lab} | {n} | {pct:.2f}% |")
    return "\n".join(lines)

def pick_examples(records: List[Dict[str, Any]], label: str, k: int = 2) -> List[Dict[str, Any]]:
    out = []
    for r in records:
        if r.get("primary") == label:
            out.append(r)
            if len(out) >= k:
                break
    return out

def generate_report(set1: List[Dict[str, Any]], set2: List[Dict[str, Any]], out_dir: Path) -> None:
    set1c = count_labels(set1)
    set2c = count_labels(set2)

    lines = []
    lines += ["# Failure Analysis v2: Direct vs CoT RAG", ""]
    lines += ["## Taxonomy (Primary labels)", ""]
    for lab in PRIMARY_LABELS:
        lines.append(f"- **{lab}**: {LABEL_PRETTY[lab]}")
    lines += ["", "## Overview", ""]
    lines += [f"- **SET1 (Direct wrong, CoT correct)**: {len(set1)}"]
    lines += [f"- **SET2 (Both wrong)**: {len(set2)}", ""]

    lines += ["## SET1 Distribution", ""]
    lines += [markdown_table(set1c, len(set1)), ""]

    lines += ["## SET2 Distribution", ""]
    lines += [markdown_table(set2c, len(set2)), ""]

    # Examples
    lines += ["## Representative Examples", ""]
    for lab in PRIMARY_LABELS:
        exs1 = pick_examples(set1, lab, 1)
        exs2 = pick_examples(set2, lab, 1)
        if not exs1 and not exs2:
            continue
        lines += [f"### {lab} — {LABEL_PRETTY[lab]}", ""]
        for ex in exs1:
            lines += [f"**SET1 | QID {ex['question_id']}**"]
            q = ex.get("question", "")
            lines += [f"- Q: {q[:220]}{'...' if len(q)>220 else ''}"]
            lines += [f"- primary={ex.get('primary')}  secondary={ex.get('secondary')}  rescue={ex.get('rescue_mechanism','') }"]
            # quotes
            dq = ex.get("audit", {}).get("direct", {}).get("gold_support", [])[:1]
            if dq:
                lines += [f"- Direct gold quote: [{dq[0].get('doc_id')}] {dq[0].get('quote')}"]
            lines += [""]

        for ex in exs2:
            lines += [f"**SET2 | QID {ex['question_id']}**"]
            q = ex.get("question", "")
            lines += [f"- Q: {q[:220]}{'...' if len(q)>220 else ''}"]
            lines += [f"- primary={ex.get('primary')}  secondary={ex.get('secondary')}"]
            dq = ex.get("audit", {}).get("direct", {}).get("gold_support", [])[:1]
            if dq:
                lines += [f"- Direct gold quote: [{dq[0].get('doc_id')}] {dq[0].get('quote')}"]
            lines += [""]

    write_text(out_dir / "paper_report_v2.md", "\n".join(lines))


# -------------------------
# Main analysis
# -------------------------

async def analyze_case(
    qid: str,
    direct_item: Dict[str, Any],
    cot_item: Dict[str, Any],
    case_type: str,
    args,
    api_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    question = to_str(direct_item.get("question", "")).strip()
    options = normalize_options(direct_item.get("options", {}))
    gold = get_correct_answer(direct_item)
    direct_pred = (get_predicted_answer(direct_item, "direct") or "").strip()
    cot_pred = (get_predicted_answer(cot_item, "cot") or "").strip()

    direct_docs = get_retrieved_docs(direct_item, "direct", args.top_docs, args.doc_max_chars)
    cot_docs = get_retrieved_docs(cot_item, "cot", args.top_docs, args.doc_max_chars)

    audit = await evidence_audit(
        question=question,
        options=options,
        gold=gold,
        direct_pred=direct_pred,
        cot_pred=cot_pred,
        direct_docs=direct_docs,
        cot_docs=cot_docs,
        model=args.audit_model,
        api_key=api_key,
        session=session,
        semaphore=semaphore,
        debug=args.debug,
    )

    # If audit failed
    if "_error" in audit:
        return {
            "question_id": qid,
            "case_type": case_type,
            "question": question,
            "gold": gold,
            "direct_pred": direct_pred,
            "cot_pred": cot_pred,
            "primary": "ERROR",
            "secondary": None,
            "rescue_mechanism": None,
            "audit": audit,
        }

    ambig = bool((audit.get("ambiguity") or {}).get("is_ambiguous", False))

    direct_blk = audit.get("direct") or {}
    cot_blk = audit.get("cot") or {}

    d_sig = method_signals(direct_blk)
    c_sig = method_signals(cot_blk)

    # Assign labels depending on set
    if case_type == "SET1":
        # direct is wrong -> label direct failure
        primary = label_wrong_prediction(d_sig, ambig, noise_thr=args.noise_thr)
        secondary = None

        # rescue mechanism (cot is correct)
        cot_correct_without_gold = (c_sig.gold_support_n == 0)
        rescue = rescue_mechanism(primary, d_sig, c_sig, cot_correct_without_gold)
        return {
            "question_id": qid,
            "case_type": case_type,
            "question": question,
            "gold": gold,
            "direct_pred": direct_pred,
            "cot_pred": cot_pred,
            "primary": primary,
            "secondary": secondary,
            "rescue_mechanism": rescue,
            "audit": audit,
        }

    else:  # SET2 both wrong
        d_lab = label_wrong_prediction(d_sig, ambig, noise_thr=args.noise_thr)
        c_lab = label_wrong_prediction(c_sig, ambig, noise_thr=args.noise_thr)
        primary = combine_set2_label(d_lab, c_lab)
        secondary = f"D={d_lab}|C={c_lab}"
        return {
            "question_id": qid,
            "case_type": case_type,
            "question": question,
            "gold": gold,
            "direct_pred": direct_pred,
            "cot_pred": cot_pred,
            "primary": primary,
            "secondary": secondary,
            "rescue_mechanism": None,
            "audit": audit,
        }

async def run_set(
    cases: List[Tuple[str, Dict[str, Any], Dict[str, Any]]],
    case_type: str,
    args,
    api_key: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    out_jsonl: Path,
) -> List[Dict[str, Any]]:
    def relabel_from_audit(record: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(record)
        audit = out.get("audit", {})
        if not isinstance(audit, dict) or "_error" in audit:
            out["primary"] = "ERROR"
            out["secondary"] = out.get("secondary")
            return out

        ambig = bool((audit.get("ambiguity") or {}).get("is_ambiguous", False))
        d_sig = method_signals(audit.get("direct") or {})
        c_sig = method_signals(audit.get("cot") or {})

        if case_type == "SET1":
            primary = label_wrong_prediction(d_sig, ambig, noise_thr=args.noise_thr)
            cot_correct_without_gold = (c_sig.gold_support_n == 0)
            rescue = rescue_mechanism(primary, d_sig, c_sig, cot_correct_without_gold)
            out["primary"] = primary
            out["secondary"] = None
            out["rescue_mechanism"] = rescue
            return out

        d_lab = label_wrong_prediction(d_sig, ambig, noise_thr=args.noise_thr)
        c_lab = label_wrong_prediction(c_sig, ambig, noise_thr=args.noise_thr)
        out["primary"] = combine_set2_label(d_lab, c_lab)
        out["secondary"] = f"D={d_lab}|C={c_lab}"
        out["rescue_mechanism"] = None
        return out

    # resume support
    done_raw = {} if args.rerun_all else load_jsonl(out_jsonl)
    done = {qid: relabel_from_audit(rec) for qid, rec in done_raw.items()}
    results: List[Dict[str, Any]] = list(done.values())

    pending = []
    for (qid, d, c) in cases:
        if (not args.rerun_all) and (qid in done):
            continue
        pending.append((qid, d, c))

    print(f"[{case_type}] total={len(cases)} pending={len(pending)} resume_skipped={len(cases)-len(pending)}")

    async def worker(qid, d, c):
        return await analyze_case(qid, d, c, case_type, args, api_key, session, semaphore)

    tasks = [asyncio.create_task(worker(qid, d, c)) for (qid, d, c) in pending]
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        append_jsonl(out_jsonl, r)

    # dedup
    uniq = {}
    for r in results:
        uniq[to_str(r.get("question_id")).strip()] = r
    return list(uniq.values())

def build_sets(direct: Dict[str, Any], cot: Dict[str, Any]) -> Tuple[List[Tuple[str, Any, Any]], List[Tuple[str, Any, Any]], int]:
    common = sorted(set(direct.keys()) & set(cot.keys()))
    set1, set2 = [], []
    skipped = 0
    for qid in common:
        dc = get_correctness(direct[qid], "direct")
        cc = get_correctness(cot[qid], "cot")
        if dc is None or cc is None:
            skipped += 1
            continue
        if (dc is False) and (cc is True):
            set1.append((qid, direct[qid], cot[qid]))
        elif (dc is False) and (cc is False):
            set2.append((qid, direct[qid], cot[qid]))
    return set1, set2, skipped

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--direct", type=Path, required=True)
    p.add_argument("--cot", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)

    # controls
    p.add_argument("--top_docs", type=int, default=8)
    p.add_argument("--doc_max_chars", type=int, default=1200)
    p.add_argument("--audit_model", type=str, default="gpt-4o")
    p.add_argument("--max_concurrent", type=int, default=100)
    p.add_argument("--noise_thr", type=float, default=0.60)
    p.add_argument("--probe_api", action="store_true")
    p.add_argument("--rerun_all", action="store_true")
    p.add_argument("--debug", action="store_true", default=False)
    return p.parse_args()

async def async_main():
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY env var.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    direct = load_json_file(args.direct)
    cot = load_json_file(args.cot)

    set1, set2, skipped = build_sets(direct, cot)
    print(f"SET1={len(set1)} SET2={len(set2)} skipped={skipped}")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async with aiohttp.ClientSession() as session:
        if args.probe_api:
            print("Probing API...")
            await probe_api(session, semaphore, api_key, args.audit_model, debug=args.debug)
            print("API OK.")

        set1_out = args.out_dir / "set1_v2.jsonl"
        set2_out = args.out_dir / "set2_v2.jsonl"

        set1_res = await run_set(set1, "SET1", args, api_key, session, semaphore, set1_out)
        set2_res = await run_set(set2, "SET2", args, api_key, session, semaphore, set2_out)

    # Summaries
    def write_summary(records: List[Dict[str, Any]], name: str):
        counts = defaultdict(int)
        for r in records:
            counts[r.get("primary", "ERROR")] += 1
        lines = [name, markdown_table(dict(counts), len(records))]
        write_text(args.out_dir / f"{name.lower()}_summary_v2.md", "\n\n".join(lines))

    write_summary(set1_res, "SET1")
    write_summary(set2_res, "SET2")
    generate_report(set1_res, set2_res, args.out_dir)

    print("Done. Outputs:", args.out_dir)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
