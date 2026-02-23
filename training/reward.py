#!/usr/bin/env python3
"""
Reward Functions for GRPO Rewriter Training

TRL-compatible reward function that:
1. Parses 3 search queries from the rewriter's completion text
2. Retrieves documents using the retriever
3. Generates an answer using a frozen generator (vLLM)
4. Computes reward by comparing predicted answer to gold

Reward types:
- acc: binary accuracy (1.0 if correct, 0.0 otherwise)
"""

import os
import sys
import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ============================================================================
# Query Parsing
# ============================================================================
def parse_queries_from_completion(completion_text: str, num_queries: int = 3) -> List[str]:
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
    pattern = r'Query\s*\d+\s*:\s*(.+?)(?=Query\s*\d+\s*:|$)'
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
            m = re.match(r'^[\d]+[.):\-]\s*(.+)$', line)
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


def parse_answer(answer_text: str) -> str:
    """Parse answer choice (A, B, C, D) from generator output.

    Reuses the same logic as evaluate_medqa.py parse_answer().
    """
    if not answer_text:
        return ""

    answer_text = str(answer_text)

    # Try JSON parsing first
    try:
        if "{" in answer_text and "}" in answer_text:
            start = answer_text.find("{")
            end = answer_text.rfind("}") + 1
            json_str = answer_text[start:end]
            answer_data = json.loads(json_str)
            choice = answer_data.get("answer_choice", answer_data.get("answer", ""))
            if choice and choice.upper() in ["A", "B", "C", "D"]:
                return choice.upper()
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex patterns
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

    # Last capital letter match
    matches = re.findall(r'\b([ABCD])\b', answer_upper)
    if matches:
        return matches[-1]

    return ""


# ============================================================================
# Generator prompt (matches MIRAGE format from evaluate_medqa.py)
# ============================================================================
MIRAGE_SYSTEM_PROMPT = (
    'You are a helpful medical expert, and your task is to answer a multi-choice medical question '
    'using the relevant documents. Please first think step-by-step and then choose the answer from '
    'the provided options. Organize your output in a json formatted as '
    'Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. '
    'Your responses will be used for research purposes only, so please have a definite answer.'
)

# Try to load the actual MIRAGE prompt
try:
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "..", "MIRAGE", "MedRAG", "src"))
    from template import general_medrag_system
    MIRAGE_SYSTEM_PROMPT = general_medrag_system
except ImportError:
    pass


def format_generator_prompt(
    question: str,
    options: Dict[str, str],
    retrieved_docs: List[Dict[str, Any]],
) -> str:
    """Format the generator prompt (docs + question + options) for vLLM.

    Isolation style: only docs + question + options, no plan or queries.
    """
    context_parts = []
    for idx, doc in enumerate(retrieved_docs[:25]):
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        context_parts.append(f"Document [{idx + 1}] (Title: {title})\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "No documents."
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])

    return f"""Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options_text}

Please think step-by-step and generate your output in json:"""


# ============================================================================
# Reward Function Class (TRL-compatible)
# ============================================================================
class RewriterRewardFunction:
    """TRL-compatible reward function for GRPO rewriter training.

    For each rewriter completion (3 queries):
        1. Parse queries from completion text
        2. Retrieve documents using the retriever
        3. Generate answer using frozen generator (vLLM)
        4. Compare with gold answer → reward

    The __call__ signature matches TRL's reward function interface:
        reward_fn(prompts, completions, **kwargs) -> list[float]

    where completions is a list of completion strings (or list of message dicts).
    """

    def __init__(
        self,
        generator_model_name: str,
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        total_docs: int = 15,
        reward_type: str = "acc",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 8192,
        retrieval_cache: Optional[Dict[str, Any]] = None,
        generator_temperature: float = 0.0,
    ):
        """
        Args:
            generator_model_name: HF model name/path for frozen generator.
            retriever_name: Retriever name (e.g., "MedCPT").
            corpus_name: Corpus name (e.g., "Textbooks").
            total_docs: Total docs to retrieve per answer.
            reward_type: "acc" for binary accuracy.
            tensor_parallel_size: vLLM tensor parallelism for generator.
            gpu_memory_utilization: vLLM GPU memory fraction for generator.
            max_model_len: vLLM max model length for generator.
            retrieval_cache: Optional shared cache dict for retrieval results.
            generator_temperature: Sampling temperature for the frozen generator (default: 0.0).
        """
        self.reward_type = reward_type
        self.total_docs = total_docs
        self.generator_temperature = generator_temperature
        self.retrieval_cache = retrieval_cache if retrieval_cache is not None else {}
        self.parse_failures = 0
        self.total_calls = 0
        self.__name__ = self.__class__.__name__

        # Initialize retriever
        from retriever import create_retriever
        self.retriever = create_retriever(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
        print(f"✓ Reward: retriever={retriever_name}, corpus={corpus_name}")

        # Initialize frozen generator via OpenAI API
        import openai
        self.generator_llm = openai.OpenAI(
            base_url="http://localhost:18081/v1",
            api_key="empty",
        )
        self.generator_model_name = generator_model_name
        print(f"✓ Reward: frozen generator={generator_model_name} loaded via OpenAI API")

    def _retrieve_documents(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve and fuse documents from multiple queries (with caching)."""
        # Cache key
        cache_key = hashlib.md5("|".join(sorted(queries)).encode()).hexdigest()
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]

        doc_scores = {}
        doc_data = {}

        k_per_query = max(1, self.total_docs // len(queries)) if queries else self.total_docs

        for query in queries:
            try:
                docs, scores = self.retriever.retrieve(query, k=k_per_query)
                for doc, score in zip(docs, scores):
                    doc_id = doc.get("id", doc.get("title", str(hash(doc.get("content", "")[:100]))))
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                        doc_data[doc_id] = doc.copy()
                    doc_scores[doc_id] += score
            except Exception:
                pass

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        retrieved_docs = []
        for doc_id in sorted_ids[:25]:
            doc = doc_data[doc_id]
            doc["fused_score"] = doc_scores[doc_id]
            retrieved_docs.append(doc)

        self.retrieval_cache[cache_key] = retrieved_docs
        return retrieved_docs

    def _generate_answer(
        self,
        question: str,
        options: Dict[str, str],
        retrieved_docs: List[Dict[str, Any]],
    ) -> str:
        """Generate answer using frozen generator (single prompt, OpenAI API)."""
        MAX_OUTPUT_TOKENS = 2048
        MAX_MODEL_LEN = 8192
        MAX_INPUT_TOKENS = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS  # 6144

        # Try with all docs first, then progressively remove docs if input is too long
        docs_to_use = list(retrieved_docs)
        while docs_to_use:
            user_prompt = format_generator_prompt(question, options, docs_to_use)
            # Rough token estimate: ~4 chars per token for English medical text
            estimated_tokens = len(user_prompt) // 3
            if estimated_tokens <= MAX_INPUT_TOKENS:
                break
            # Remove the last (lowest-scored) document and retry
            docs_to_use = docs_to_use[:-1]

        if not docs_to_use:
            user_prompt = format_generator_prompt(question, options, [])

        messages = [
            {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.generator_llm.chat.completions.create(
                model=self.generator_model_name,
                messages=messages,
                temperature=self.generator_temperature,
                max_tokens=MAX_OUTPUT_TOKENS,
                stop=["###", "User:", "\n\n\n"],
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Warning: Reward generation failed: {e}")
            return ""

    def _generate_answers_batch(
        self,
        questions: List[str],
        options_list: List[Dict[str, str]],
        docs_list: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """Generate answers for a batch of questions using frozen generator (OpenAI API ThreadPool)."""
        from concurrent.futures import ThreadPoolExecutor
        
        def _call_api(args):
            return self._generate_answer(*args)
            
        with ThreadPoolExecutor(max_workers=min(16, len(questions) + 1)) as executor:
            args_list = list(zip(questions, options_list, docs_list))
            return list(executor.map(_call_api, args_list))

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        gold_answer: Optional[List[str]] = None,
        question: Optional[List[str]] = None,
        options: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """Compute rewards for a batch of completions.

        TRL calls this with: reward_fn(prompts, completions, **extra_columns)
        where extra columns are from the dataset (gold_answer, question, options, etc.)

        Args:
            prompts: List of prompt strings.
            completions: List of completion strings (rewriter outputs).
            gold_answer: List of gold answer letters (from dataset).
            question: List of original question texts (from dataset).
            options: List of options JSON strings (from dataset).

        Returns:
            List of float rewards.
        """
        self.total_calls += len(completions)

        # Extract completion text
        completion_texts = []
        for c in completions:
            if isinstance(c, list):
                # TRL may pass as list of message dicts
                text = c[-1]["content"] if c else ""
            elif isinstance(c, dict):
                text = c.get("content", str(c))
            else:
                text = str(c)
            completion_texts.append(text)

        # Parse queries for each completion
        all_queries = []
        for text in completion_texts:
            queries = parse_queries_from_completion(text)
            if len(queries) < 2:
                self.parse_failures += 1
            all_queries.append(queries)

        # Retrieve documents for each set of queries
        all_docs = []
        for queries in all_queries:
            if queries:
                docs = self._retrieve_documents(queries)
            else:
                docs = []
            all_docs.append(docs)

        # Parse options from JSON strings
        options_dicts = []
        if options is not None:
            for opt_str in options:
                if isinstance(opt_str, str):
                    try:
                        options_dicts.append(json.loads(opt_str))
                    except (json.JSONDecodeError, ValueError):
                        options_dicts.append({})
                elif isinstance(opt_str, dict):
                    options_dicts.append(opt_str)
                else:
                    options_dicts.append({})
        else:
            options_dicts = [{}] * len(completions)

        # Extract question texts
        question_texts = question if question is not None else [""] * len(completions)

        # Generate answers in batch using vLLM
        raw_answers = self._generate_answers_batch(question_texts, options_dicts, all_docs)

        # Compute rewards
        rewards = []
        gold_answers = gold_answer if gold_answer is not None else [""] * len(completions)
        for raw_ans, gold in zip(raw_answers, gold_answers):
            predicted = parse_answer(raw_ans)
            if self.reward_type == "acc":
                reward = 1.0 if predicted.upper() == gold.upper() else 0.0
            else:
                reward = 1.0 if predicted.upper() == gold.upper() else 0.0
            rewards.append(reward)

        return rewards

    def get_metrics(self) -> Dict[str, Any]:
        """Get accumulated metrics."""
        return {
            "total_calls": self.total_calls,
            "parse_failures": self.parse_failures,
            "parse_failure_rate": (
                self.parse_failures / self.total_calls if self.total_calls > 0 else 0.0
            ),
            "retrieval_cache_size": len(self.retrieval_cache),
        }
