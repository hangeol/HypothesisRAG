#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test JSON parsing logic to verify structure compatibility
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

def to_str(x: Any) -> str:
    return "" if x is None else str(x)

def load_json_file(file_path: Path) -> Dict[str, Any]:
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
    for item in rows:
        qid = item.get("question_id") or item.get("id") or item.get("qid")
        qid = to_str(qid).strip()
        if qid:
            out[qid] = item
    return out

def get_mode_block(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    modes = item.get("modes", {})
    if isinstance(modes, dict) and mode in modes and isinstance(modes[mode], dict):
        return modes[mode]
    return {}

def get_correct_answer(item: Dict[str, Any]) -> str:
    return to_str(item.get("correct_answer")).strip()

def get_predicted_answer(item: Dict[str, Any], mode: str) -> Optional[str]:
    m = get_mode_block(item, mode)
    for k in ["predicted_answer", "prediction", "answer_choice", "pred"]:
        if k in m and m[k] is not None:
            return to_str(m[k]).strip()

    raw = m.get("raw_response")
    if isinstance(raw, str) and raw.strip().startswith("{"):
        try:
            j = json.loads(raw)
            for k in ["answer_choice", "predicted_answer"]:
                if k in j:
                    return to_str(j[k]).strip()
        except Exception:
            pass
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

def get_retrieved_docs(item: Dict[str, Any], mode: str, top_n: int, max_chars: Optional[int]) -> List[Dict[str, Any]]:
    m = get_mode_block(item, mode)
    docs = m.get("retrieved_docs", []) or []
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
            "content": to_str(content)[:max_chars] if max_chars else to_str(content),
        })
    return out

def normalize_options(raw_opts: Any) -> Dict[str, str]:
    if raw_opts is None:
        return {}
    if isinstance(raw_opts, dict):
        return {to_str(k).strip(): to_str(v) for k, v in raw_opts.items()}
    if isinstance(raw_opts, list):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        out = {}
        for i, v in enumerate(raw_opts):
            k = letters[i] if i < len(letters) else str(i)
            out[k] = to_str(v)
        return out
    return {}

def main():
    direct_path = Path("results/medqa_direct_full.json")
    cot_path = Path("results/medqa_cot_full.json")
    
    print("=" * 80)
    print("Testing JSON Parsing Logic")
    print("=" * 80)
    
    # Load files
    print("\n1. Loading JSON files...")
    try:
        direct = load_json_file(direct_path)
        cot = load_json_file(cot_path)
        print(f"   ✓ Direct: {len(direct)} items")
        print(f"   ✓ CoT: {len(cot)} items")
    except Exception as e:
        print(f"   ✗ Error loading files: {e}")
        return
    
    # Check common question IDs
    common = sorted(set(direct.keys()) & set(cot.keys()))
    print(f"\n2. Common question IDs: {len(common)}")
    
    if not common:
        print("   ✗ No common question IDs found!")
        return
    
    # Test parsing on first few items
    print("\n3. Testing parsing logic on first 5 items...")
    issues = []
    
    for i, qid in enumerate(common[:5]):
        print(f"\n   Item {i+1}: QID={qid}")
        d_item = direct[qid]
        c_item = cot[qid]
        
        # Test question_id extraction
        d_qid = to_str(d_item.get("question_id") or d_item.get("id") or d_item.get("qid")).strip()
        c_qid = to_str(c_item.get("question_id") or c_item.get("id") or c_item.get("qid")).strip()
        if d_qid != qid or c_qid != qid:
            issues.append(f"QID {qid}: question_id mismatch")
            print(f"      ✗ question_id mismatch: direct={d_qid}, cot={c_qid}")
        else:
            print(f"      ✓ question_id: {qid}")
        
        # Test correct_answer
        gold = get_correct_answer(d_item)
        if not gold:
            issues.append(f"QID {qid}: missing correct_answer")
            print(f"      ✗ missing correct_answer")
        else:
            print(f"      ✓ correct_answer: {gold}")
        
        # Test options
        opts = normalize_options(d_item.get("options", {}))
        if not opts:
            issues.append(f"QID {qid}: missing or empty options")
            print(f"      ✗ missing options")
        else:
            print(f"      ✓ options: {list(opts.keys())}")
        
        # Test mode blocks
        d_mode = get_mode_block(d_item, "direct")
        c_mode = get_mode_block(c_item, "cot")
        if not d_mode:
            issues.append(f"QID {qid}: missing direct mode")
            print(f"      ✗ missing direct mode")
        else:
            print(f"      ✓ direct mode: {list(d_mode.keys())[:5]}...")
        if not c_mode:
            issues.append(f"QID {qid}: missing cot mode")
            print(f"      ✗ missing cot mode")
        else:
            print(f"      ✓ cot mode: {list(c_mode.keys())[:5]}...")
        
        # Test predicted answers
        d_pred = get_predicted_answer(d_item, "direct")
        c_pred = get_predicted_answer(c_item, "cot")
        print(f"      ✓ direct_pred: {d_pred}")
        print(f"      ✓ cot_pred: {c_pred}")
        
        # Test correctness
        d_correct = get_correctness(d_item, "direct")
        c_correct = get_correctness(c_item, "cot")
        print(f"      ✓ direct_correct: {d_correct}")
        print(f"      ✓ cot_correct: {c_correct}")
        
        # Test retrieved docs
        d_docs = get_retrieved_docs(d_item, "direct", top_n=5, max_chars=100)
        c_docs = get_retrieved_docs(c_item, "cot", top_n=5, max_chars=100)
        print(f"      ✓ direct_docs: {len(d_docs)} docs")
        print(f"      ✓ cot_docs: {len(c_docs)} docs")
        
        if d_docs:
            doc_keys = list(d_docs[0].keys())
            print(f"      ✓ direct_doc structure: {doc_keys}")
        if c_docs:
            doc_keys = list(c_docs[0].keys())
            print(f"      ✓ cot_doc structure: {doc_keys}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if issues:
        print(f"✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All parsing logic tests passed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
