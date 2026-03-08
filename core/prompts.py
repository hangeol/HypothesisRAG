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

HYPOTHESIS_V7_USER_PROMPT = (
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
)

HYPOTHESIS_V7PLUS_USER_PROMPT = (
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Self-confidence scale for your final best_guess:\n"
    "1 = not confident (high chance of being wrong)\n"
    "2 = somewhat confident (ambiguous)\n"
    "3 = very confident (almost certain)\n"
    "Use 3 only when you are strongly certain. "
    "If torn between options, use 1 or 2.\n\n"
    "Analyze this question carefully. Think step-by-step about "
    "each option, considering the presentation and "
    "relevant knowledge. Reason through the elimination analysis "
    "before making your final assessment.\n\n"
    "Return exactly one JSON object in this format:\n"
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
    '    "alternative_if_wrong": "A/B/C/D - only if uncertain",\n'
    '    "confidence_level": "<1|2|3>"\n'
    '}}\n\n'
    "confidence_level must be an integer in {{1,2,3}} only."
)

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
        "system":"You are an expert analyst taking an exam.",
        "user": HYPOTHESIS_V7_USER_PROMPT,
        "description": "hv5 + best_guess_text field in JSON",
    },
    "v8": {
        "system":"You are an expert analyst taking an exam.",#"You are an expert question-answering assistant.",
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
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "best_guess_text": '
            '"<<<copy the chosen option text verbatim>>>",\n'
            '    "closest_alternative": "A/B/C/D",\n'
            '    "closest_alternative_text": '
            '"<<<copy the closest alternative option text verbatim>>>"\n'
            '}}'
        ),
        "description": "hv5 + best_guess_text field in JSON",
    }, "v9": {
        "system":"You are an expert analyst taking an exam.",#"You are an expert question-answering assistant.",
        "user": (
            "Question: {question}\n\n"
            "Options:\n{options}\n\n"
            "Analyze this question carefully. Think step-by-step about "
            "each option, considering the information given in the question and "
            "relevant knowledge. Reason through the elimination analysis "
            "before making your final assessment.\n\n"
            "After your analysis, provide your final assessment in JSON:\n"
            '{{\n'
            '    "discriminating_features": '
            '["2-3 features that distinguish between options"],\n'
            '    "reasoning": '
            '"brief explanation why this is the best answer",\n'
            '    "confirming_evidence": '
            '["1-3 specific facts that would confirm this answer"],\n'
            '    "best_guess": "A/B/C/D",\n'
            '    "best_guess_text": '
            '"<<<copy the chosen option text verbatim>>>",\n'
            '}}'
        ),
        "description": "hv5 + best_guess_text field in JSON",
    },
    "v7_sys_medrag": {
        "system": (
            "You are a board-certified physician answering a high-stakes "
            "medical exam question. Be clinically precise and avoid overclaiming."
        ),
        "user": HYPOTHESIS_V7_USER_PROMPT,
        "description": "v7 with MedRAG-style clinical expert tone",
    },
    "v7_sys_calibrated": {
        "system": (
            "You are an expert medical reasoner. Avoid overconfidence and make "
            "conservative conclusions when evidence is mixed."
        ),
        "user": HYPOTHESIS_V7_USER_PROMPT,
        "description": "v7 with calibrated certainty style",
    },
    "v7_sys_diffdx": {
        "system": (
            "You are an expert diagnostician. Prioritize differential diagnosis, "
            "explicitly weighing strongest support and strongest contradiction."
        ),
        "user": HYPOTHESIS_V7_USER_PROMPT,
        "description": "v7 with differential-diagnosis focused reasoning style",
    },
    "v7_sys_strict_json": {
        "system": (
            "You are an expert analyst. Return structured, faithful outputs only. "
            "Do not include extra text outside the required JSON."
        ),
        "user": HYPOTHESIS_V7_USER_PROMPT,
        "description": "v7 with strict formatting emphasis",
    },
    "v7plus": {
        "system": (
            "You are an expert analyst "
            "taking an exam."
        ),
        "user": HYPOTHESIS_V7PLUS_USER_PROMPT,
        "description": "v7 + self-reported confidence_level(1/2/3) for gating",
    },
    "v7plus_sys_medrag": {
        "system": (
            "You are a board-certified physician answering a high-stakes "
            "medical exam question. Be clinically precise and avoid overclaiming."
        ),
        "user": HYPOTHESIS_V7PLUS_USER_PROMPT,
        "description": "v7plus with MedRAG-style clinical expert tone",
    },
    "v7plus_sys_calibrated": {
        "system": (
            "You are an expert medical reasoner. Calibrate confidence strictly: "
            "reserve very high confidence for clear textbook presentations."
        ),
        "user": HYPOTHESIS_V7PLUS_USER_PROMPT,
        "description": "v7plus with explicit confidence calibration emphasis",
    },
    "v7plus_sys_diffdx": {
        "system": (
            "You are an expert diagnostician. Prioritize differential diagnosis, "
            "explicitly weighing strongest support and strongest contradiction."
        ),
        "user": HYPOTHESIS_V7PLUS_USER_PROMPT,
        "description": "v7plus with differential-diagnosis focused reasoning style",
    },
    "v7plus_sys_strict_json": {
        "system": (
            "You are an expert analyst. Return structured, faithful outputs only. "
            "Do not inflate certainty when evidence is mixed."
        ),
        "user": HYPOTHESIS_V7PLUS_USER_PROMPT,
        "description": "v7plus with strict formatting + uncertainty honesty",
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
    "v11": {
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
            "Closest Alternative: {closest_alternative_text}\n"
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
    "v12": {
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
            "Closest Alternative: {closest_alternative_text}\n"
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
MEDRAG_SYSTEM_PROMPT = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

COT_USER_PROMPT_TEMPLATE = '''
Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
'''

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
# v3 format variables: {context}, {question}, {options},
#                       {hypothesis_summary}, {queries_summary},
#                       {best_guess_text}, {closest_alternative_text}

GENERATOR_PROMPTS = {
    "v1": {
        "system": MEDRAG_SYSTEM_PROMPT,
        "user":MEDRAG_USER_PROMPT_TEMPLATE,
        "description": "Standard MIRAGE generator",
    },
    "v2": {
        "system": MEDRAG_SYSTEM_PROMPT,
        "user": (
            "Here is the question:\n{question}\n\n"
            "Here are the potential choices:\n{options}\n\n"
            "Diagnostic Hypothesis:\n{hypothesis_summary}\n\n"
            "Search Queries Used:\n{queries_summary}\n\n"
            "Here are the relevant documents:\n{context}\n\n"
            "Consider the diagnostic hypothesis and retrieved evidence. "
            "Think step-by-step and generate your output in json:"
        ),
        "description": "MIRAGE generator with hypothesis context",
    },
    "v3": {
        "system": MEDRAG_SYSTEM_PROMPT,
        "user": (
            "Here is the question:\n{question}\n\n"
            "Here are the potential choices:\n{options}\n\n"
            "Primary Hypothesis Candidate:\n{best_guess_text}\n\n"
            "Closest Alternative Candidate:\n{closest_alternative_text}\n\n"
            "Diagnostic Hypothesis:\n{hypothesis_summary}\n\n"
            "Search Queries Used:\n{queries_summary}\n\n"
            "Here are the relevant documents:\n{context}\n\n"
            "Use retrieved documents as primary evidence. "
            "Treat hypothesis and closest alternative as reference signals only. "
            "If document evidence conflicts with those references, follow the documents. "
            "Think step-by-step and generate your output in json:"
        ),
        "description": "MIRAGE generator with hypothesis + closest alternative context",
    },
}


# ============================================================================
# Trust Evaluator / Hypothesis Finalizer Prompts
# ============================================================================

TRUST_EVALUATOR_PROMPTS = {
    "v1": {
        "system": (
            "You are a strict medical QA auditor. Evaluate only whether the given "
            "hypothesis JSON is reliable for final answering."
        ),
        "user": (
            "Question:\n{question}\n\n"
            "Options:\n{options}\n\n"
            "Hypothesis JSON:\n{hypothesis_json}\n\n"
            "Assess whether this hypothesis is trustworthy.\n"
            "Return exactly one JSON object:\n"
            '{{\n'
            '  "hypothesis_trust": 1,\n'
            '  "trust_reason": "short reason",\n'
            '  "risk_flags": ["up to 3 concise risks"]\n'
            '}}\n\n'
            "Scale:\n"
            "1 = low trust (likely wrong/incomplete)\n"
            "2 = medium trust (usable but uncertain)\n"
            "3 = high trust (coherent and likely correct)\n"
            "hypothesis_trust must be an integer in {{1,2,3}} only."
        ),
        "description": "Trust evaluator over full hypothesis JSON",
    },
}


HYPOTHESIS_FINALIZER_PROMPTS = {
    "v1": {
        "system": (
            "You are a medical expert. Answer a multi-choice medical question "
            "using only the provided hypothesis JSON and question/options. "
            "Return JSON with answer_choice."
        ),
        "user": (
            "Question:\n{question}\n\n"
            "Options:\n{options}\n\n"
            "Hypothesis JSON:\n{hypothesis_json}\n\n"
            "Using only this hypothesis, choose the best final answer.\n"
            'Return JSON: {{"step_by_step_thinking": "...", "answer_choice": "A/B/C/D"}}'
        ),
        "description": "No-retrieval finalizer conditioned on hypothesis JSON",
    },
}


# ============================================================================
# Baseline Prompts (for non-hypothesis modes: cot, directrag, directrewriting)
# ============================================================================


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
        "trust_evaluator": TRUST_EVALUATOR_PROMPTS,
        "hypothesis_finalizer": HYPOTHESIS_FINALIZER_PROMPTS,
        "system": {
            "cot": COT_SYSTEM_PROMPT,
            "medrag": MEDRAG_SYSTEM_PROMPT,
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
