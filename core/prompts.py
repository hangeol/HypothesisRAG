#!/usr/bin/env python3
"""
Prompt definitions for HypothesisRAG evaluation.

All prompts are organized by module (hypothesis, rewriting, generator)
with version variants (v1, v2, v3) for ablation studies.

Each prompt is a dict with:
  - "system": system prompt string
  - "user":   user prompt template (with {format_variables})
  - "description": brief description for logging/paper reference
"""

import ast
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ============================================================================
# Hypothesis Prompts (Phase 1: Diagnostic Plan Generation)
# ============================================================================
# Format variables: {question}, {options}

HYPOTHESIS_PROMPTS = {
    "v1": {
        "system": (
            "You are an expert medical diagnostician "
            "taking a medical licensing exam."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Step 1: Identify the KEY DISCRIMINATING FEATURES "
            "that distinguish between the options.\n"
            "Step 2: Make your BEST GUESS for the answer "
            "based on medical knowledge.\n"
            "Step 3: Identify what SPECIFIC EVIDENCE "
            "would CONFIRM your answer.\n\n"
            "Output in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "alternative_if_wrong": "A/B/C/D - only if uncertain"\n'
            '}}'
        ),
        "description": "Structured plan (GRPO training prompt)",
    },
    "v2": {
        "system": (
            "You are an expert medical diagnostician "
            "taking a medical licensing exam."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering the clinical presentation and "
            "relevant medical knowledge. Reason through the differential "
            "diagnosis before making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "alternative_if_wrong": "A/B/C/D - only if uncertain"\n'
            '}}'
        ),
        "description": "COT + structured plan (think before JSON)",
    },
    "v3": {
        "system": (
            "You are an expert diagnostician "
            "taking a licensing exam."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering the presentation and "
            "relevant knowledge. Reason through the differential "
            "diagnosis before making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "alternative_if_wrong": "A/B/C/D - only if uncertain"\n'
            '}}'
        ),
        "description": "COT + structured plan (no medical terms)",
    },
    "v4": {
        "system": (
            "You are an expert problem solver. "
            "Analyze the question and provide your best answer."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering all relevant information and "
            "key distinctions. Reason through your analysis before "
            "making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "alternative_if_wrong": "A/B/C/D - only if uncertain"\n'
            '}}'
        ),
        "description": "Universal COT + structured plan (domain-agnostic)",
    },
    "v5": {
        "system": (
            "You are an expert analyst "
            "taking an exam."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering the presentation and "
            "relevant knowledge. Reason through the elimination analysis "
            "before making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "alternative_if_wrong": "A/B/C/D - only if uncertain"\n'
            '}}'
        ),
        "description": "Universal (hv3 structure, domain-neutral wording)",
    },
    "v6": {
        "system": (
            "You are an expert problem solver. "
            "Analyze the question and provide your best answer."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering the presentation and "
            "relevant knowledge. Reason through the elimination analysis "
            "before making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "<your chosen option>",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"]\n'
            '}}'
        ),
        "description": "Universal v6 (no exam/ABCD hints, no alternative)",
    },
    "v7": {
        "system": (
            "You are an expert analyst "
            "taking an exam."
        ),
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering the presentation and "
            "relevant knowledge. Reason through the elimination analysis "
            "before making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "best_guess_text": '
            '"<<<copy the chosen option text verbatim>>>",\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "alternative_if_wrong": "A/B/C/D - only if uncertain"\n'
            '}}'
        ),
        "description": "hv5 + best_guess_text field in JSON",
    },
}


# ============================================================================
# Rewriting Prompts (Phase 2: Query Generation)
# ============================================================================
# Format variables: {question}, {best_guess}, {reasoning},
# {confirming_evidence}, {discriminating_features}, {alternative_if_wrong}

REWRITING_PROMPTS = {
    "v1": {
        "system": (
            "You are a medical search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this medical question.\n\n"
            "Question: {question}\n"
            "Best Guess Answer: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "for differential diagnosis\n"
            "Query 3: Find specific clinical/pathological features\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "Targeted medical queries (3 specific)",
    },
    "v2": {
        "system": (
            "You are a medical search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this medical question.\n\n"
            "Question: {question}\n"
            "Best Guess Answer: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n"
            "Alternative If Wrong: {alternative_if_wrong}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria between "
            "{best_guess} and {alternative_if_wrong}\n"
            "Query 3: Find specific clinical/pathological features\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "Targeted medical queries (with alternative comparison)",
    },
    "v3": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate exactly 3 targeted, non-overlapping search queries "
            "to find evidence for this question.\n\n"
            "Question: {question}\n"
            "Best Guess: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n"
            "Alternative If Wrong: {alternative_if_wrong}\n\n"
            "Generate the 3 best, non-overlapping queries. "
            "Each query must target different information "
            "to maximize evidence coverage.\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "Non-overlapping queries (no domain-specific terms)",
    },
    "v4": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate exactly 3 targeted, non-overlapping search queries "
            "to find evidence for this question.\n\n"
            "Question: {question}\n"
            "Best Guess: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n"
            "Alternative If Wrong: {alternative_if_wrong}\n\n"
            "Generate the 3 best, non-overlapping queries:\n"
            "Query 1: Find evidence supporting {best_guess}\n"
            "Query 2: Find distinguishing criteria between "
            "{best_guess} and {alternative_if_wrong}\n"
            "Query 3: Find key features or facts that confirm "
            "the correct answer\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "Universal non-overlapping queries (domain-agnostic)",
    },
    "v5": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this question.\n\n"
            "Question: {question}\n"
            "Best Guess Answer: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "between the top candidate answers\n"
            "Query 3: Find specific key features or facts\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "Universal (rv1 structure, domain-neutral wording)",
    },
    "v6": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this question.\n\n"
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Best Guess Answer: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "between the top candidate answers\n"
            "Query 3: Find specific key features or facts\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "Universal v6 (with options, no alternative)",
    },
    "v7": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this question.\n\n"
            "Question: {question}\n"
            "Best Guess: {best_guess_text}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess_text} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "between the top candidate answers\n"
            "Query 3: Find specific key features or facts\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "rv5 + expanded best_guess text (no all-options, no alternative)",
    },
    "v8": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this question.\n\n"
            "Question: {question}\n"
            "Best Guess Answer: {best_guess}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n"
            "Alternative If Wrong: {alternative_if_wrong}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "between {best_guess} and {alternative_if_wrong}\n"
            "Query 3: Find specific key features or facts\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "rv5 + alternative_if_wrong (contrastive query)",
    },
    "v9": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this question.\n\n"
            "Question: {question}\n"
            "Best Guess: {best_guess_text}\n"
            "Alternative If Wrong: {alternative_text}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess_text} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "between {best_guess_text} and {alternative_text}\n"
            "Query 3: Find specific key features or facts\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "rv7 + expanded alternative text (full contrastive)",
    },
    "v10": {
        "system": (
            "You are a search query expert. "
            "Generate precise, targeted search queries. "
            "Output ONLY the 3 queries in the exact format requested."
        ),
        "user": (
            "Generate 3 highly targeted search queries to find evidence "
            "for this question.\n\n"
            "Question: {question}\n"
            "Best Guess Answer: {best_guess_text}\n"
            "Reasoning: {reasoning}\n"
            "Evidence Needed: {confirming_evidence}\n"
            "Key Features: {discriminating_features}\n\n"
            "Generate 3 SPECIFIC queries:\n"
            "Query 1: Find evidence supporting {best_guess_text} "
            "- focus on the main reasoning\n"
            "Query 2: Find distinguishing criteria "
            "between the top candidate answers\n"
            "Query 3: Find specific key features or facts\n\n"
            "Format:\n"
            "Query 1: [query]\n"
            "Query 2: [query]\n"
            "Query 3: [query]"
        ),
        "description": "rv5 with best_guess_text replacing best_guess",
    },
}



COT_SYSTEM_PROMPT = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''


MIRAGE_SYSTEM_PROMPT = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''


# CoT user message template.
COT_USER_PROMPT_TEMPLATE = '''
Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
'''

# MedRAG user message template (used by directrag/directrewriting and generator v1 style).
MEDRAG_USER_PROMPT_TEMPLATE = '''
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
'''

# Multi-query prompt for baseline mode (3 sub-queries)
MULTI_QUERY_PROMPT_3 = """You are an AI language model assistant. Your task\
 is to generate exactly three different versions of the\
 given user question to retrieve relevant documents\
 from a vector database. By generating multiple\
 perspectives on the user question, your goal is to\
 help the user overcome some of the limitations of\
 the distance-based similarity search.

Original question: {query}

Format your response in plain text as:

Sub-query 1:

Sub-query 2:

Sub-query 3:
"""
#from MMLF: Multi-query Multi-passage Late Fusion Retrieval
# Clear alias used by evaluate.py for direct rewriting mode.
DIRECT_REWRITING_PROMPT = MULTI_QUERY_PROMPT_3
DIRECT_REWRITING_SYSTEM_PROMPT = ""
DIRECT_REWRITING_TARGET_QUERIES = 3
DIRECT_REWRITING_DOCS_PER_QUERY = 5

# Simple planning prompt for 'planning' baseline mode
PLANNING_PROMPT = """Analyze the following medical question and extract key information.

Question: {question}

Extract:
1. observed_features: List 3-7 key symptoms, findings, or conditions mentioned
2. must_check_cooccurrence: List pairs of features whose co-occurrence is important
3. need_disambiguation: List any confusing concepts that need distinction

Output in JSON format:
{{"observed_features": [...], "must_check_cooccurrence": [[...]], "need_disambiguation": [...]}}"""










# ============================================================================
# Generator Prompts (Phase 4: Final Answer Generation)
# ============================================================================
# v1 format variables: {context}, {question}, {options}
# v2 format variables: {context}, {question}, {options},
#                       {hypothesis_summary}, {queries_summary}

GENERATOR_PROMPTS = {
    "v1": {
        "system": MIRAGE_SYSTEM_PROMPT,
        "user":MEDRAG_USER_PROMPT_TEMPLATE,
        "description": "Standard MIRAGE generator",
    },
    "v2": {
        "system": (
            'You are a helpful medical expert, and your task is to answer '
            'a multi-choice medical question using the relevant documents '
            'and diagnostic hypothesis. '
            'Please first think step-by-step and then choose the answer '
            'from the provided options. Organize your output in a json '
            'formatted as Dict{"step_by_step_thinking": Str(explanation), '
            '"answer_choice": Str{A/B/C/...}}. Your responses will be '
            'used for research purposes only, so please have a definite answer.'
        ),
        "user": (
            "Diagnostic Hypothesis:\n{hypothesis_summary}\n\n"
            "Search Queries Used:\n{queries_summary}\n\n"
            "Here are the relevant documents:\n{context}\n\n"
            "Here is the question:\n{question}\n\n"
            "Here are the potential choices:\n{options}\n\n"
            "Consider the diagnostic hypothesis and retrieved evidence. "
            "Think step-by-step and generate your output in json:"
        ),
        "description": "MIRAGE generator with hypothesis context",
    },
}


# ============================================================================
# Baseline Prompts (for non-hypothesis modes: cot, directrag, directrewriting)
# ============================================================================







# ============================================================================
# Helper: Load MIRAGE template prompts (override defaults if available)
# ============================================================================
def _find_mirage_template_path() -> Optional[Path]:
    """Return MIRAGE template.py path if available in this project."""
    project_root = Path(__file__).resolve().parents[1]
    template_path = project_root / "MIRAGE" / "MedRAG" / "src" / "template.py"
    if template_path.exists():
        return template_path
    return None


def _read_mirage_system_prompts() -> Optional[Tuple[str, str]]:
    """
    Load MIRAGE system prompts from template.py without mutating sys.path.

    Returns:
        (general_medrag_system, general_cot_system) if available, else None.
    """
    template_path = _find_mirage_template_path()
    if template_path is None:
        return None

    loaded_from_import = _read_mirage_system_prompts_via_import(template_path)
    if loaded_from_import is not None:
        return loaded_from_import
    return _read_mirage_system_prompts_via_source(template_path)


def _read_mirage_system_prompts_via_import(
    template_path: Path,
) -> Optional[Tuple[str, str]]:
    """Load prompts by importing template.py."""
    spec = importlib_util.spec_from_file_location(
        "hypothesisrag_mirage_template",
        str(template_path),
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib_util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    medrag_system = getattr(module, "general_medrag_system", None)
    cot_system = getattr(module, "general_cot_system", None)
    if isinstance(medrag_system, str) and isinstance(cot_system, str):
        return medrag_system, cot_system
    return None


def _read_mirage_system_prompts_via_source(
    template_path: Path,
) -> Optional[Tuple[str, str]]:
    """
    Fallback path for environments missing template.py dependencies.

    We only need two constant strings, so parse assignment nodes directly.
    """
    try:
        source = template_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(template_path))
    except Exception:
        return None

    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"general_medrag_system", "general_cot_system"}:
            continue

        value_node = node.value
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            values[target.id] = value_node.value
        elif isinstance(value_node, ast.Str):
            values[target.id] = value_node.s

    medrag_system = values.get("general_medrag_system")
    cot_system = values.get("general_cot_system")
    if isinstance(medrag_system, str) and isinstance(cot_system, str):
        return medrag_system, cot_system
    return None


def load_mirage_prompts() -> bool:
    """
    Override built-in system prompts with MIRAGE's canonical template prompts.

    Why this exists:
      - Keeps CoT and generator system prompts aligned with
        MIRAGE/MedRAG/src/template.py.
      - Prevents drift between copied local prompt text and upstream MIRAGE.
    """
    global COT_SYSTEM_PROMPT, MIRAGE_SYSTEM_PROMPT

    loaded = _read_mirage_system_prompts()
    if loaded is None:
        return False

    medrag_system, cot_system = loaded
    MIRAGE_SYSTEM_PROMPT = medrag_system
    COT_SYSTEM_PROMPT = cot_system

    # Keep generator v1/v2 system prompts aligned with MIRAGE.
    GENERATOR_PROMPTS["v1"]["system"] = medrag_system
    GENERATOR_PROMPTS["v2"]["system"] = medrag_system
    return True


def get_evaluate_prompt_bundle() -> Dict[str, Any]:
    """
    Single prompt bundle for evaluate.py.

    Call this after load_mirage_prompts() so system prompts reflect MIRAGE
    overrides when available.
    """
    return {
        "hypothesis": HYPOTHESIS_PROMPTS,
        "rewriter": REWRITING_PROMPTS,
        "generator": GENERATOR_PROMPTS,
        "system": {
            "cot": COT_SYSTEM_PROMPT,
            "medrag": MIRAGE_SYSTEM_PROMPT,
        },
        "answer": {
            "cot_user": COT_USER_PROMPT_TEMPLATE,
            "medrag_user": MEDRAG_USER_PROMPT_TEMPLATE,
        },
        "baseline": {
            "direct_rewriting_system": DIRECT_REWRITING_SYSTEM_PROMPT,
            "direct_rewriting_user": DIRECT_REWRITING_PROMPT,
            "direct_rewriting_num_queries": DIRECT_REWRITING_TARGET_QUERIES,
            "direct_rewriting_docs_per_query": DIRECT_REWRITING_DOCS_PER_QUERY,
            "planning_user": PLANNING_PROMPT,
        },
    }
