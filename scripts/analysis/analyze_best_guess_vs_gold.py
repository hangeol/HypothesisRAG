import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Counter

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
        qid = str(item.get("question_id") or item.get("id") or item.get("qid") or i).strip()
        if qid:
            out[qid] = item
    return out

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    data = []
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return data

def main():
    parser = argparse.ArgumentParser(description="Analyze Best Guess vs Gold by Failure Category")
    parser.add_argument("--run_dir", type=Path, required=True, help="Path to the timestamped run directory (e.g. results/run_XXX)")
    parser.add_argument("--target", type=Path, required=True, help="Path to the original target JSON file containing planning_v4 data")
    args = parser.parse_args()

    # Load original target data to get best_guess
    print(f"Loading target data from {args.target}...")
    target_data = load_json_file(args.target)
    
    # Load analysis results
    set1_path = args.run_dir / "set1_v2.jsonl"
    set2_path = args.run_dir / "set2_v2.jsonl"
    
    set1_cases = load_jsonl(set1_path)
    set2_cases = load_jsonl(set2_path)
    
    print(f"Loaded {len(set1_cases)} SET1 cases and {len(set2_cases)} SET2 cases.")
    
    all_cases = set1_cases + set2_cases
    
    # Stats storage: Category -> {total, match, mismatch}
    stats = {}

    missing_best_guess = 0
    
    for case in all_cases:
        qid = str(case.get("question_id")).strip()
        category = case.get("primary", "UNKNOWN")
        gold = case.get("gold", "").strip()
        
        # Get best guess from target data
        best_guess = None
        if qid in target_data:
            item = target_data[qid]
            # Try to find planning_v4 or any mode with plan/best_guess
            modes = item.get("modes", {})
            # Look for keys like "planning_v4" or "cot" or just first avail
            # Based on user input, we expect "planning_v4"
            if "planning_v4" in modes:
                plan = modes["planning_v4"].get("plan", {})
                best_guess = plan.get("best_guess")
            # If not found, try generic search
            if not best_guess:
                for m_key in modes:
                    plan = modes[m_key].get("plan", {})
                    if "best_guess" in plan:
                        best_guess = plan["best_guess"]
                        break
        
        if not best_guess:
            # Fallback output check? 
            # Sometimes best guess is just the prediction if not explicitly in plan
            # But user specifically asked for "plan's best guess".
            missing_best_guess += 1
            continue
            
        best_guess = best_guess.strip()

        if category not in stats:
            stats[category] = {"total": 0, "match": 0, "mismatch": 0}
        
        stats[category]["total"] += 1
        if best_guess == gold:
            stats[category]["match"] += 1
        else:
            stats[category]["mismatch"] += 1

    print(f"\nAnalysis Report (Cases with valid Best Guess):")
    print("-" * 80)
    print(f"{'Category':<15} | {'Total':<8} | {'Match (Gold==BG)':<18} | {'Mismatch (Gold!=BG)':<18}")
    print("-" * 80)
    
    for cat in sorted(stats.keys()):
        s = stats[cat]
        total = s["total"]
        match = s["match"]
        mismatch = s["mismatch"]
        match_pct = (match / total) * 100 if total > 0 else 0
        mismatch_pct = (mismatch / total) * 100 if total > 0 else 0
        
        print(f"{cat:<15} | {total:<8} | {match:<4} ({match_pct:6.2f}%)   | {mismatch:<4} ({mismatch_pct:6.2f}%)")
        
    print("-" * 80)
    if missing_best_guess > 0:
        print(f"Warning: {missing_best_guess} cases had no 'best_guess' in the target file.")

if __name__ == "__main__":
    main()
