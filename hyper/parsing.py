"""Parse hypothesis plans, evidence queries, and answer choices."""
import re
import json
from typing import Any, Dict, List, Optional


def parse_queries_from_completion(
    completion_text: str, num_queries: int = 3
) -> List[str]:
    """Parse Query 1/2/3 from rewriter completion text.

    Handles slight format drift (e.g., missing colons, extra whitespace).

    Args:
        completion_text: Raw completion text from the rewriter.
        num_queries: Expected number of queries.

    Returns:
        List of parsed query strings. May be shorter than num_queries if parsing fails.
    """
    queries = []

    # Pattern: "Query N: <text>"  (flexible whitespace, optional brackets)
    pattern = r"Query\s*\d+\s*:\s*(.+?)(?=Query\s*\d+\s*:|$)"
    matches = re.findall(pattern, completion_text, re.IGNORECASE | re.DOTALL)

    for match in matches:
        query = match.strip().strip("[]\"'").strip()
        if query and len(query) > 5:
            queries.append(query)

    # Fallback: try numbered list "1. <text>" or "1) <text>"
    if len(queries) < 2:
        queries = []
        for line in completion_text.strip().split("\n"):
            line = line.strip()
            m = re.match(r"^[\d]+[.):\-]\s*(.+)$", line)
            if m:
                q = m.group(1).strip().strip("[]\"'").strip()
                if q and len(q) > 5:
                    queries.append(q)

    # Final fallback: split by newlines, take first N non-empty lines
    if len(queries) < 2:
        queries = []
        for line in completion_text.strip().split("\n"):
            line = line.strip()
            if line and len(line) > 10 and not line.lower().startswith("query"):
                queries.append(line)
            if len(queries) >= num_queries:
                break

    return queries[:num_queries]


def _coerce_str_list(value: Any) -> List[str]:
    """Normalize list-like fields from plan JSON."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _extract_first_json_object(answer_text: str) -> Optional[Dict[str, Any]]:
    """Extract first valid JSON object from arbitrary text."""
    text = str(answer_text)
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, n):
            ch = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        parsed = json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def _coerce_choice_letter(value: Any) -> str:
    """Normalize answer choice into A/B/C/D."""
    if value is None:
        return ""
    text = str(value).upper().strip()
    match = re.findall(r"\b([ABCD])\b", text)
    return match[-1] if match else ""


def _coerce_confidence_level(value: Any, default: int = 2) -> int:
    """Normalize confidence into integer 1/2/3."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value in (1, 2, 3) else default
    text = str(value).strip()
    if not text:
        return default
    m = re.search(r"\b([123])\b", text)
    if m:
        return int(m.group(1))
    return default


def parse_hypothesis_plan(answer_text: str) -> Dict[str, Any]:
    """Parse hypothesis / plan output from hypothesis model completion."""
    if not answer_text:
        answer_text = ""

    text = str(answer_text)
    plan_data = _extract_first_json_object(text) or {}

    # Regex fallback when JSON parsing fails
    if not plan_data:
        best_guess = None
        match = re.search(
            r'"?best[_\s-]*guess"?\s*[:=]\s*["\']?([ABCD])["\']?',
            text,
            re.IGNORECASE,
        )
        if match:
            best_guess = match.group(1).upper()
        if not best_guess:
            match = re.search(
                r"\bBEST\s*GUESS\b[^A-D]{0,40}\b([ABCD])\b", text, re.IGNORECASE
            )
            if match:
                best_guess = match.group(1).upper()
        plan_data = {"best_guess": best_guess or ""}

    plan: Dict[str, Any] = {
        "discriminating_features": _coerce_str_list(
            plan_data.get("discriminating_features", [])
        ),
        "best_guess": _coerce_choice_letter(plan_data.get("best_guess", "")),
        "best_guess_text": str(plan_data.get("best_guess_text", "") or "").strip(),
        "reasoning": str(plan_data.get("reasoning", "") or "").strip(),
        "confirming_evidence": _coerce_str_list(
            plan_data.get("confirming_evidence", [])
        ),
        "alternative_if_wrong": _coerce_choice_letter(
            plan_data.get("alternative_if_wrong", "")
        ),
        "confidence_level": _coerce_confidence_level(
            plan_data.get("confidence_level", 2),
            default=2,
        ),
    }
    return plan


def parse_answer(answer_text: str) -> str:
    """Parse answer choice (A, B, C, D) from model output"""
    if not answer_text:
        return ""

    answer_text = str(answer_text)

    try:
        if "{" in answer_text and "}" in answer_text:
            start = answer_text.find("{")
            end = answer_text.rfind("}") + 1
            json_str = answer_text[start:end]
            answer_data = json.loads(json_str)
            choice = answer_data.get("answer_choice", answer_data.get("answer", ""))
            if choice and choice.upper() in ["A", "B", "C", "D"]:
                return choice.upper()
    except:
        pass

    answer_upper = answer_text.upper()
    patterns = [
        r"ANSWER IS[:\s]*([ABCD])",
        r"ANSWER[:\s]*([ABCD])",
        r"CHOICE[:\s]*([ABCD])",
        r"\"ANSWER_CHOICE\"[:\s]*\"([ABCD])\"",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer_upper)
        if match:
            return match.group(1)

    matches = re.findall(r"\b([ABCD])\b", answer_upper)
    if matches:
        return matches[-1]

    return ""


def coerce_choice(value: Any) -> str:
    text = str(value or "").upper()
    match = re.findall(r"\b([ABCD])\b", text)
    return match[-1] if match else ""


def coerce_confidence_level(value: Any, default: int = 2) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value in (1, 2, 3) else default
    m = re.search(r"\b([123])\b", str(value or ""))
    if m:
        return int(m.group(1))
    return default


def _resolve_choice_text(
    options: Dict[str, str],
    letter: str,
    explicit_text: str = "",
) -> str:
    """Resolve option letter to `A. option text`, preferring explicit text when provided."""
    if explicit_text and str(explicit_text).strip():
        return str(explicit_text).strip()
    choice = coerce_choice(letter)
    if choice and choice in options:
        return f"{choice}. {options.get(choice, choice)}"
    return choice or ""


def _enrich_plan_with_alternatives(
    plan: Dict[str, Any],
    raw_text: str,
    options: Dict[str, str],
) -> Dict[str, Any]:
    """Resolve hypothesis and alternative option labels from generated text."""
    extra = _extract_first_json_object(raw_text) or {}

    # Recover the selected option text when omitted.
    bg_letter = coerce_choice(plan.get("best_guess", "") or extra.get("best_guess", ""))
    plan["best_guess"] = bg_letter
    plan["best_guess_text"] = _resolve_choice_text(
        options,
        bg_letter,
        str(
            plan.get("best_guess_text", "") or extra.get("best_guess_text", "")
        ).strip(),
    )

    # Closest alternative may be emitted as either key.
    closest_letter = coerce_choice(
        plan.get("closest_alternative", "")
        or plan.get("alternative_if_wrong", "")
        or extra.get("closest_alternative", "")
        or extra.get("alternative_if_wrong", "")
    )
    closest_text_raw = str(
        plan.get("closest_alternative_text", "")
        or extra.get("closest_alternative_text", "")
    ).strip()
    closest_text = _resolve_choice_text(options, closest_letter, closest_text_raw)

    plan["closest_alternative"] = closest_letter
    plan["closest_alternative_text"] = closest_text
    # Keep the alternative option fields aligned.
    plan["alternative_if_wrong"] = closest_letter

    return plan
