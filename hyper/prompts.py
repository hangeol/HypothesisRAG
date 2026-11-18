"""Prompts for hypothesis formation, evidence-seeking queries, and final answers."""

HYPOTHESIS_PROMPT = {
    "system": """You are an expert analyst taking an exam.""",
    "user": """Question: {question}

Options:
{options}

Analyze this question carefully. Think step-by-step about each option, considering the information given in the question and relevant knowledge. Reason through the elimination analysis before making your final assessment.

After your analysis, provide your final assessment in JSON:
{{
    "discriminating_features": ["2-3 features that distinguish between options"],
    "reasoning": "brief explanation why this is the best answer",
    "confirming_evidence": ["1-3 specific facts that would confirm this answer"],
    "best_guess": "A/B/C/D",
    "best_guess_text": "<<<copy the chosen option text verbatim>>>",
}}""",
}

QUERY_PROMPT = {
    "system": """You are a search query expert. Generate precise, targeted search queries. Output ONLY the 3 queries in the exact format requested.""",
    "user": """Generate 3 highly targeted search queries to find evidence for this question.

Question: {question}
Best Guess Answer: {best_guess_text}
Reasoning: {reasoning}
Evidence Needed: {confirming_evidence}
Key Features: {discriminating_features}

Generate 3 SPECIFIC queries:
Query 1: Find evidence supporting {best_guess_text} - focus on the main reasoning
Query 2: Find distinguishing criteria between the top candidate answers
Query 3: Find specific key features or facts

Format:
Query 1: [query]
Query 2: [query]
Query 3: [query]""",
}

ANSWER_PROMPT = {
    "system": """You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.""",
    "user": """
Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options}

Please think step-by-step and generate your output in json:
""",
}
