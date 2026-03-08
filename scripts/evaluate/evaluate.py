#!/usr/bin/env python3
"""
MedQA Evaluation v2 – Prompt-Selectable Ablation System

Modes:
  cot       : Chain-of-Thought (no RAG)
  directrag : 1 query × total_docs
  directrewriting: 3 sub-queries × 5 docs/query (=15 docs)
  hypothesis: Hypothesis→Rewrite→Retrieve→Answer (4-phase batch)

For hypothesis mode, select prompt versions:
  --hypothesis-prompt  v1 | v2 | v3
  --rewriting-prompt   v1 | v2 | v3
  --generator-prompt   v1 | v2 | v3
  --run-all  runs all 9 hypothesis×rewriting combos (generator=v1)
"""

import os
import sys
import json
import time
import re
import math
import asyncio
import argparse
import random
import csv
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import aiohttp

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "MIRAGE"),
    os.path.join(PROJECT_ROOT, "MIRAGE", "MedRAG"),
    os.path.join(PROJECT_ROOT, "MIRAGE", "MedRAG", "src"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval.retriever import create_retriever
from core.prompts import (
    get_evaluate_prompt_bundle,
)


RETRIEVAL_DATASET_TO_CORPUS = {
    "textbooks": "Textbooks",
    "pubmed": "PubMed",
}
CORPUS_TO_RETRIEVAL_DATASET = {
    corpus.lower(): dataset for dataset, corpus in RETRIEVAL_DATASET_TO_CORPUS.items()
}


def parse_retrieval_dataset(value: str) -> str:
    """Normalize retrieval dataset input to canonical MIRAGE corpus name."""
    normalized = value.strip().lower()
    if normalized not in RETRIEVAL_DATASET_TO_CORPUS:
        allowed = ", ".join(sorted(RETRIEVAL_DATASET_TO_CORPUS.keys()))
        raise argparse.ArgumentTypeError(
            f"Invalid retrieval dataset '{value}'. Choose one of: {allowed}"
        )
    return RETRIEVAL_DATASET_TO_CORPUS[normalized]


QUESTION_DATASET_CHOICES = ("medqa", "pubmedqa", "medmcqa", "bioasq", "mmlu")


def parse_question_dataset(value: str) -> str:
    """Normalize question dataset input to benchmark key."""
    normalized = value.strip().lower()
    if normalized not in QUESTION_DATASET_CHOICES:
        allowed = ", ".join(QUESTION_DATASET_CHOICES)
        raise argparse.ArgumentTypeError(
            f"Invalid question dataset '{value}'. Choose one of: {allowed}"
        )
    return normalized


def _slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return value or "model"


def resolve_results_output_dir(
    output_dir: Optional[str],
    llm_provider: str,
    model_name: str,
) -> str:
    """Auto-route results under outputs/results/<provider>/<model>/ if not specified."""
    if output_dir:
        return output_dir
    provider = "openai" if llm_provider == "openai" else "local"
    return os.path.join(
        PROJECT_ROOT,
        "outputs",
        "results",
        provider,
        _slugify(model_name),
    )


def resolve_ablation_summary_dir(
    output_dir: Optional[str],
    llm_provider: str,
) -> str:
    if output_dir:
        return output_dir
    provider = "openai" if llm_provider == "openai" else "local"
    return os.path.join(PROJECT_ROOT, "outputs", "results", provider, "ablation")


def parse_ablation_combo_tokens(
    tokens: List[str],
    default_generator_prompt: str,
) -> List[Tuple[str, str, str]]:
    combos: List[Tuple[str, str, str]] = []
    for token in tokens:
        parts = token.split("-")
        if len(parts) == 2:
            hv, rv = parts
            gv = default_generator_prompt
        elif len(parts) == 3:
            hv, rv, gv = parts
        else:
            raise ValueError(
                f"Invalid combo '{token}'. Use 'vH-vR' or 'vH-vR-vG' "
                f"(e.g., v5-v5 or v7-v10-v2)."
            )
        if hv not in HYPOTHESIS_PROMPTS:
            raise ValueError(f"Unknown hypothesis prompt: {hv}")
        if rv not in REWRITING_PROMPTS:
            raise ValueError(f"Unknown rewriting prompt: {rv}")
        if gv not in GENERATOR_PROMPTS:
            raise ValueError(f"Unknown generator prompt: {gv}")
        combos.append((hv, rv, gv))
    return combos


def resolve_openai_api_key() -> Optional[str]:
    """Resolve OPENAI_API_KEY from env, with ~/.bashrc fallback."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    bashrc_path = os.path.expanduser("~/.bashrc")
    if not os.path.exists(bashrc_path):
        return None

    try:
        text = open(bashrc_path, "r", encoding="utf-8").read()
    except Exception:
        return None

    patterns = [
        r'export\s+OPENAI_API_KEY\s*=\s*"([^"]+)"',
        r"export\s+OPENAI_API_KEY\s*=\s*'([^']+)'",
        r"export\s+OPENAI_API_KEY\s*=\s*([^\s#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


# Resolve all prompt resources from a single bundle.
PROMPT_BUNDLE = get_evaluate_prompt_bundle()
HYPOTHESIS_PROMPTS = PROMPT_BUNDLE["hypothesis"]
REWRITING_PROMPTS = PROMPT_BUNDLE["rewriter"]
GENERATOR_PROMPTS = PROMPT_BUNDLE["generator"]
TRUST_EVALUATOR_PROMPTS = PROMPT_BUNDLE["trust_evaluator"]
HYPOTHESIS_FINALIZER_PROMPTS = PROMPT_BUNDLE["hypothesis_finalizer"]
COT_SYSTEM_PROMPT = PROMPT_BUNDLE["system"]["cot"]
MIRAGE_SYSTEM_PROMPT = PROMPT_BUNDLE["system"]["medrag"]
COT_USER_PROMPT_TEMPLATE = PROMPT_BUNDLE["answer"]["cot_user"]
MEDRAG_USER_PROMPT_TEMPLATE = PROMPT_BUNDLE["answer"]["medrag_user"]
DIRECT_REWRITING_SYSTEM_PROMPT = PROMPT_BUNDLE["baseline"]["direct_rewriting_system"]
DIRECT_REWRITING_PROMPT = PROMPT_BUNDLE["baseline"]["direct_rewriting_user"]
DIRECT_REWRITING_TARGET_QUERIES = int(
    PROMPT_BUNDLE["baseline"]["direct_rewriting_num_queries"]
)
DIRECT_REWRITING_DOCS_PER_QUERY = int(
    PROMPT_BUNDLE["baseline"]["direct_rewriting_docs_per_query"]
)
DIRECT_REWRITING_TOTAL_DOCS = (
    DIRECT_REWRITING_TARGET_QUERIES * DIRECT_REWRITING_DOCS_PER_QUERY
)
PLANNING_PROMPT = PROMPT_BUNDLE["baseline"]["planning_user"]


MODE_ALIASES = {
    "direct": "directrag",
    "baseline": "directrewriting",
}

EVAL_FIXED_TEMPERATURE = 0.0
EVAL_FIXED_SEED = 42
EVAL_FIXED_DO_SAMPLE = False


def normalize_mode_name(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def normalize_mode_list(modes: List[str]) -> List[str]:
    seen = set()
    normalized = []
    for mode in modes:
        mapped = normalize_mode_name(mode)
        if mapped not in seen:
            seen.add(mapped)
            normalized.append(mapped)
    return normalized

def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility setup across common libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


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


def should_use_retrieval(confidence_level: int, gating_policy: str) -> bool:
    """Return whether to use retrieval path under the selected gating policy."""
    conf = coerce_confidence_level(confidence_level, default=2)
    if gating_policy == "conf3_skip":
        return conf in (1, 2)
    if gating_policy == "conf23_skip":
        return conf == 1
    if gating_policy == "conf3_only_retrieve":
        return conf == 3
    raise ValueError(f"Unknown gating policy: {gating_policy}")


def is_high_trust(trust_level: int, trust_threshold: int) -> bool:
    """Return whether trust evaluator score is considered high-trust."""
    level = coerce_confidence_level(trust_level, default=2)
    threshold = coerce_confidence_level(trust_threshold, default=3)
    return level >= threshold


def _retrieve_fused_docs_for_query_set(
    retriever,
    query_set: List[str],
    total_docs: int,
) -> List[Dict[str, Any]]:
    """Retrieve docs for one query set and fuse by summed score."""
    if not retriever:
        return []
    doc_scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict[str, Any]] = {}
    if not query_set:
        return []

    k_per = max(1, total_docs // max(len(query_set), 1))
    for query in query_set:
        try:
            docs, scores = retriever.retrieve(query, k=k_per)
            for doc, score in zip(docs, scores):
                doc_id = doc.get(
                    "id",
                    doc.get("title", str(hash(doc.get("content", "")[:100]))),
                )
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_data[doc_id] = doc.copy()
                    doc_data[doc_id]["query_trace"] = []
                try:
                    doc_scores[doc_id] += float(score)
                except Exception:
                    doc_scores[doc_id] += 0.0
                doc_data[doc_id]["query_trace"].append(query)
        except Exception:
            pass

    docs_sorted = []
    for doc_id in sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True):
        doc = doc_data[doc_id]
        doc["fused_score"] = doc_scores[doc_id]
        docs_sorted.append(doc)
    return docs_sorted


def _to_json_safe(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            out[str(k)] = _to_json_safe(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _to_json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _serialize_retrieved_docs_for_output(
    docs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep retrieval evidence in output JSON with stable, safe fields."""
    export_keys = [
        "id",
        "doc_id",
        "chunk_id",
        "title",
        "source",
        "content",
        "text",
        "score",
        "fused_score",
        "query_trace",
        "metadata",
    ]
    serialized: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        if not isinstance(doc, dict):
            serialized.append(
                {
                    "id": f"doc_{idx+1}",
                    "content": _to_json_safe(doc),
                }
            )
            continue

        row: Dict[str, Any] = {}
        for key in export_keys:
            if key in doc:
                row[key] = _to_json_safe(doc.get(key))

        if "id" not in row:
            row["id"] = str(
                doc.get("id")
                or doc.get("chunk_id")
                or doc.get("doc_id")
                or f"doc_{idx+1}"
            )
        if "title" not in row and doc.get("title") is not None:
            row["title"] = _to_json_safe(doc.get("title"))

        if "content" not in row:
            if doc.get("content") is not None:
                row["content"] = _to_json_safe(doc.get("content"))
            elif doc.get("text") is not None:
                row["content"] = _to_json_safe(doc.get("text"))
            else:
                row["content"] = ""

        serialized.append(row)
    return serialized


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# ============================================================================
# Benchmark Dataset Loader
# ============================================================================
class MedQADataset:
    """benchmark.json dataset loader"""
    
    def __init__(self, benchmark_path: Optional[str] = None, question_dataset: str = "medqa"):
        if benchmark_path is None:
            possible_paths = [
                os.path.join(PROJECT_ROOT, "MIRAGE", "benchmark.json"),
                os.path.join(PROJECT_ROOT, "data", "benchmark.json"),
                "/mnt/data1/home/hangeol/project/MIRAGE/benchmark.json",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    benchmark_path = path
                    break
            if benchmark_path is None:
                raise FileNotFoundError(f"benchmark.json not found")
        
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark = json.load(f)

        if question_dataset not in benchmark:
            raise KeyError(
                f"Dataset '{question_dataset}' not found in benchmark.json. "
                f"Available: {sorted(benchmark.keys())}"
            )

        self.question_dataset = question_dataset
        self.dataset = benchmark[question_dataset]
        self.index = sorted(self.dataset.keys())
        print(f"✓ Loaded {len(self)} {question_dataset} questions")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, key) -> Dict[str, Any]:
        if isinstance(key, int):
            return self.dataset[self.index[key]]
        elif isinstance(key, slice):
            return [self.__getitem__(i) for i in range(len(self))[key]]
        else:
            raise KeyError(f"Key type {type(key)} not supported")


# ============================================================================
# Answer Parsing
# ============================================================================
def parse_answer(answer_text: str) -> str:
    """Parse answer choice (A, B, C, D) from model output"""
    if not answer_text:
        return ""
    
    answer_text = str(answer_text)
    
    try:
        if '{' in answer_text and '}' in answer_text:
            start = answer_text.find('{')
            end = answer_text.rfind('}') + 1
            json_str = answer_text[start:end]
            answer_data = json.loads(json_str)
            choice = answer_data.get('answer_choice', answer_data.get('answer', ''))
            if choice and choice.upper() in ['A', 'B', 'C', 'D']:
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
    
    matches = re.findall(r'\b([ABCD])\b', answer_upper)
    if matches:
        return matches[-1]
    
    return ""


def parse_subqueries(
    content: str,
    num_queries: int = DIRECT_REWRITING_TARGET_QUERIES,
) -> List[str]:
    """Parse sub-queries from LLM response"""
    queries = []
    
    pattern = r'Sub-query\s*\d+\s*:\s*(.+?)(?=Sub-query\s*\d+\s*:|$)'
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        query = match.strip()
        if query and len(query) > 5:
            queries.append(query)
    
    if len(queries) < 2:
        pattern2 = r'^\d+[.)]\s*(.+)$'
        for line in content.split('\n'):
            match = re.match(pattern2, line.strip())
            if match:
                query = match.group(1).strip()
                if query and len(query) > 5:
                    queries.append(query)

    return queries[:num_queries]


def _expand_queries_to_target_count(
    queries: List[str],
    question: str,
    target_count: int,
) -> List[str]:
    """Ensure deterministic query count by expanding with stable fallbacks."""
    if target_count <= 0:
        return []

    cleaned: List[str] = []
    seen = set()
    for query in queries:
        query_text = re.sub(r"\s+", " ", str(query)).strip()
        if not query_text:
            continue
        key = query_text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(query_text)
        if len(cleaned) >= target_count:
            return cleaned[:target_count]

    words = re.findall(r"\b[A-Za-z]{4,}\b", question)
    anchor = " ".join(words[:3]) if words else question
    fallback_candidates = [
        question,
        f"{anchor} symptoms diagnosis",
        f"{anchor} differential diagnosis",
        f"{anchor} diagnostic criteria",
        f"{anchor} treatment",
        f"{anchor} pathophysiology",
        f"{anchor} risk factors",
    ]

    for candidate in fallback_candidates:
        candidate_text = re.sub(r"\s+", " ", candidate).strip()
        if not candidate_text:
            continue
        key = candidate_text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(candidate_text)
        if len(cleaned) >= target_count:
            return cleaned[:target_count]

    # Absolute fallback: synthesize variants until target is met.
    base = cleaned[0] if cleaned else question
    suffixes = ["overview", "clinical features", "workup", "management", "prognosis"]
    idx = 0
    while len(cleaned) < target_count:
        candidate_text = f"{base} {suffixes[idx % len(suffixes)]}".strip()
        key = candidate_text.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(candidate_text)
        idx += 1

    return cleaned[:target_count]


def parse_plan(content: str) -> Dict[str, Any]:
    """Parse planning output from LLM response"""
    try:
        if '{' in content and '}' in content:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback
    return {
        "observed_features": [],
        "must_check_cooccurrence": [],
        "need_disambiguation": [],
    }


# ============================================================================
# Async OpenAI Client
# ============================================================================
class AsyncOpenAIClient:
    """High-performance async OpenAI client"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 100,
        max_retries: int = 3,
        api_base: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._is_local = bool(api_base)
        if api_base:
            self.base_url = api_base.rstrip('/') + '/chat/completions'
        else:
            self.base_url = "https://api.openai.com/v1/chat/completions"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        session: aiohttp.ClientSession = None,
    ) -> str:
        """Make async chat completion request"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        _ = temperature
        # GPT-5 family has stricter chat-completions params:
        # - uses max_completion_tokens instead of max_tokens
        # - temperature currently only supports default value, so omit it
        is_gpt5_family = self.model.startswith("gpt-5")
        token_param = "max_completion_tokens" if is_gpt5_family else "max_tokens"
        payload = {
            "model": self.model,
            "messages": messages,
            token_param: 2048,
        }
        if not is_gpt5_family:
            payload["temperature"] = EVAL_FIXED_TEMPERATURE

        # For local vLLM servers: disable Qwen3 thinking mode to avoid
        # extremely long <think> chains that make requests take 10+ minutes.
        if self._is_local:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        # Use longer timeout for local vLLM (model inference is slower than API)
        request_timeout = aiohttp.ClientTimeout(total=300 if self._is_local else 60)

        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=request_timeout,
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data["choices"][0]["message"]["content"]
                        elif response.status == 429:
                            # Rate limited, wait and retry
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await response.text()
                            if attempt == self.max_retries - 1:
                                return f"Error: {response.status} - {error_text[:200]}"
                            await asyncio.sleep(1)
                except asyncio.TimeoutError:
                    if attempt == self.max_retries - 1:
                        return "Error: Timeout"
                    await asyncio.sleep(1)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return f"Error: {str(e)}"
                    await asyncio.sleep(1)
        
        return "Error: Max retries exceeded"


# ============================================================================
# Async vLLM Client
# ============================================================================
class VLLMAsyncClient:
    """Async wrapper around local vLLM inference (matching MedRAG pattern)"""

    def __init__(
        self,
        model: str,
        max_concurrent: int = 1,
        max_retries: int = 1,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 2048,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
    ):
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        # Allow configurable in-process request parallelism for local vLLM.
        self.semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM backend selected but `vllm` is not installed. "
                "Install it with: pip install vllm"
            ) from e

        self._SamplingParams = SamplingParams
        print(f"Initializing vLLM for {model} (tp={tensor_parallel_size}, "
              f"max_model_len={max_model_len}, gpu_mem={gpu_memory_utilization})")
        self._llm = LLM(
            model=model,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            seed=EVAL_FIXED_SEED,
        )
        try:
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                stop=["###", "User:", "\n\n\n"],
                seed=EVAL_FIXED_SEED,
            )
        except TypeError:
            # Older vLLM versions may not expose SamplingParams(seed=...).
            self._sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                stop=["###", "User:", "\n\n\n"],
            )
        self._tokenizer = self._llm.get_tokenizer()
        # vLLM LLM.generate is not thread-safe under high concurrent to_thread calls.
        # Serialize generate() calls to avoid EngineCore socket corruption.
        self._generate_lock = threading.Lock()

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages into a model prompt for vLLM"""
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            lines = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                lines.append(f"{role}: {content}")
            lines.append("assistant:")
            return "\n".join(lines)

    def _generate_sync(self, messages: List[Dict[str, str]], temperature: float) -> str:
        prompt = self._format_prompt(messages)
        _ = temperature
        with self._generate_lock:
            outputs = self._llm.generate(
                [prompt],
                sampling_params=self._sampling_params,
                use_tqdm=True,
            )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    def _generate_batch_sync(
        self,
        messages_batch: List[List[Dict[str, str]]],
        temperature: float,
    ) -> List[str]:
        prompts = [self._format_prompt(messages) for messages in messages_batch]
        _ = temperature
        with self._generate_lock:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=True,
            )
        texts = _extract_vllm_output_texts(outputs)
        if len(texts) < len(messages_batch):
            texts.extend([""] * (len(messages_batch) - len(texts)))
        return texts[:len(messages_batch)]

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        session: aiohttp.ClientSession = None,
    ) -> str:
        """Generate text using local vLLM (session kept for API compatibility)"""
        del session
        temperature = EVAL_FIXED_TEMPERATURE
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    return await asyncio.to_thread(
                        self._generate_sync,
                        messages,
                        temperature,
                    )
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return f"Error: {str(e)}"
                    await asyncio.sleep(1)
        return "Error: Max retries exceeded"

    async def chat_completion_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        temperature: float = 0,
        session: aiohttp.ClientSession = None,
    ) -> List[str]:
        """Generate a batch of completions with one vLLM generate() call."""
        del session
        if not messages_batch:
            return []

        temperature = EVAL_FIXED_TEMPERATURE
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    return await asyncio.to_thread(
                        self._generate_batch_sync,
                        messages_batch,
                        temperature,
                    )
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return [f"Error: {str(e)}"] * len(messages_batch)
                    await asyncio.sleep(1)
        return ["Error: Max retries exceeded"] * len(messages_batch)


# ============================================================================
# Async RAG Evaluator
# ============================================================================
class AsyncRAGEvaluator:
    """Maximum performance async RAG evaluation"""
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        api_key: Optional[str] = None,
        max_concurrent: int = 100,
        num_subqueries: int = 5,
        total_docs: int = 15,
        vllm_tensor_parallel_size: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_max_tokens: int = 2048,
        vllm_max_concurrent: int = 1,
        vllm_max_model_len: int = 8192,
        rewriter_adapter_path: Optional[str] = None,
        rewriter_base_model: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.num_subqueries = num_subqueries
        self.total_docs = total_docs
        self.max_concurrent = max_concurrent
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Async LLM client (OpenAI or local vLLM)
        if self.llm_provider == "openai":
            self.client = AsyncOpenAIClient(
                api_key=self.api_key,
                model=model_name,
                max_concurrent=max_concurrent,
                api_base=api_base,
            )
        elif self.llm_provider == "vllm":
            self.client = VLLMAsyncClient(
                model=model_name,
                max_concurrent=vllm_max_concurrent,
                tensor_parallel_size=vllm_tensor_parallel_size,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                max_tokens=vllm_max_tokens,
                max_model_len=vllm_max_model_len,
            )
        else:
            raise ValueError(f"Unsupported llm_provider: {llm_provider}")

        # GRPO-trained rewriter (optional, for planning_v4_grpo mode)
        # Supports both LoRA adapters and full fine-tuned checkpoints.
        self.grpo_rewriter_model = None
        self.grpo_rewriter_tokenizer = None
        if rewriter_adapter_path:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            base_name = rewriter_base_model or model_name
            is_lora = os.path.exists(os.path.join(rewriter_adapter_path, "adapter_config.json"))

            if is_lora:
                from peft import PeftModel
                print(f"Loading GRPO rewriter LoRA adapter from {rewriter_adapter_path}...")
                self.grpo_rewriter_tokenizer = AutoTokenizer.from_pretrained(
                    base_name, trust_remote_code=True
                )
                if self.grpo_rewriter_tokenizer.pad_token is None:
                    self.grpo_rewriter_tokenizer.pad_token = self.grpo_rewriter_tokenizer.eos_token
                base_m = AutoModelForCausalLM.from_pretrained(
                    base_name, torch_dtype=torch.bfloat16,
                    device_map="auto", trust_remote_code=True,
                )
                self.grpo_rewriter_model = PeftModel.from_pretrained(
                    base_m, rewriter_adapter_path
                )
                self.grpo_rewriter_model.eval()
                print(f"✓ GRPO rewriter LoRA adapter loaded (base: {base_name})")
            else:
                print(f"Loading GRPO rewriter full checkpoint from {rewriter_adapter_path}...")
                self.grpo_rewriter_tokenizer = AutoTokenizer.from_pretrained(
                    rewriter_adapter_path, trust_remote_code=True
                )
                if self.grpo_rewriter_tokenizer.pad_token is None:
                    self.grpo_rewriter_tokenizer.pad_token = self.grpo_rewriter_tokenizer.eos_token
                self.grpo_rewriter_model = AutoModelForCausalLM.from_pretrained(
                    rewriter_adapter_path, torch_dtype=torch.bfloat16,
                    device_map="auto", trust_remote_code=True,
                )
                self.grpo_rewriter_model.eval()
                print(f"✓ GRPO rewriter full checkpoint loaded from {rewriter_adapter_path}")
        
        # Initialize retriever (sync, but fast)
        self.retriever = create_retriever(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
    
    def retrieve_documents(
        self,
        queries: List[str],
        k_per_query: int,
        max_docs: int = 25,
    ) -> List[Dict[str, Any]]:
        """Retrieve and fuse documents from multiple queries."""
        doc_scores = {}
        doc_data = {}
        
        for query in queries:
            try:
                docs, scores = self.retriever.retrieve(query, k=k_per_query)
                for doc, score in zip(docs, scores):
                    doc_id = doc.get("id", doc.get("title", str(hash(doc.get("content", "")[:100]))))
                    
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = 0.0
                        doc_data[doc_id] = doc.copy()
                        doc_data[doc_id]["query_trace"] = []
                    
                    doc_scores[doc_id] += score
                    doc_data[doc_id]["query_trace"].append(query)
            except Exception as e:
                pass
        
        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        
        retrieved_docs = []
        for doc_id in sorted_ids[:max_docs]:
            doc = doc_data[doc_id]
            doc["fused_score"] = doc_scores[doc_id]
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    async def generate_plan_async(
        self,
        question: str,
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Generate planning output asynchronously"""
        prompt = PLANNING_PROMPT.format(question=question)
        
        messages = [
            {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        plan = parse_plan(response)
        
        if not plan.get("observed_features"):
            words = re.findall(r'\b[A-Za-z]{4,}\b', question)
            plan["observed_features"] = words[:5] if words else ["symptom"]
        
        return plan
    
    async def generate_subqueries_async(
        self,
        question: str,
        plan: Optional[Dict] = None,
        session: aiohttp.ClientSession = None,
        system_prompt: Optional[str] = None,
        target_num_queries: Optional[int] = None,
    ) -> List[str]:
        """Generate subqueries asynchronously"""
        target_count = target_num_queries or self.num_subqueries

        if plan and plan.get("observed_features"):
            features = plan.get("observed_features", [])
            cooccurrence = plan.get("must_check_cooccurrence", [])
            features_str = ", ".join(features[:5]) if features else "N/A"
            cooccurrence_pairs = []
            for item in cooccurrence[:2]:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 2:
                        cooccurrence_pairs.append(f"({item[0]}&{item[1]})")
                    elif len(item) == 1:
                        cooccurrence_pairs.append(f"({item[0]})")
                else:
                    cooccurrence_pairs.append(f"({item})")
            cooccurrence_str = " & ".join(cooccurrence_pairs) if cooccurrence_pairs else ""
            plan_summary = f"Key features: {features_str}"
            if cooccurrence_str:
                plan_summary += f" | Must-check: {cooccurrence_str}"
            enhanced_query = f"User question: {question} | {plan_summary}"
        else:
            enhanced_query = question
        
        prompt = DIRECT_REWRITING_PROMPT.format(query=enhanced_query)
        used_system_prompt = MIRAGE_SYSTEM_PROMPT if system_prompt is None else system_prompt
        messages = [
            {"role": "system", "content": used_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        queries = parse_subqueries(response, target_count)
        return self._expand_queries_to_target(queries, question, target_count)

    def _expand_queries_to_target(
        self,
        queries: List[str],
        question: str,
        target_count: int,
    ) -> List[str]:
        return _expand_queries_to_target_count(queries, question, target_count)
    
    # =========================================================================
    # Planning V2: Improved diagnostic reasoning approach
    # =========================================================================
    async def generate_plan_v2_async(
        self,
        question: str,
        options: Dict[str, str],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Generate improved planning output with diagnostic reasoning"""
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        prompt = PLANNING_V2_PROMPT.format(question=question, options=options_text)
        
        messages = [
            {"role": "system", "content": "You are an expert medical diagnostician. Provide precise, clinically-focused analysis."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse the planning response
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                plan = json.loads(response[start:end])
            else:
                plan = {}
        except:
            plan = {}
        
        # Ensure required fields exist
        if not plan.get("key_clinical_features"):
            words = re.findall(r'\b[A-Za-z]{4,}\b', question)
            plan["key_clinical_features"] = words[:5] if words else ["symptom"]
        if not plan.get("primary_diagnosis_hypothesis"):
            plan["primary_diagnosis_hypothesis"] = "unknown condition"
        if not plan.get("differential_diagnoses"):
            plan["differential_diagnoses"] = []
        if not plan.get("distinguishing_features"):
            plan["distinguishing_features"] = []
        
        return plan
    
    async def generate_queries_from_plan_v2_async(
        self,
        question: str,
        plan: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> List[str]:
        """Generate targeted queries based on the diagnostic plan"""
        
        key_features = ", ".join(plan.get("key_clinical_features", [])[:5])
        primary_diagnosis = plan.get("primary_diagnosis_hypothesis", "unknown")
        differentials = ", ".join(plan.get("differential_diagnoses", [])[:3])
        distinguishing = ", ".join(plan.get("distinguishing_features", [])[:3])
        
        prompt = PLANNING_V2_QUERY_PROMPT.format(
            question=question,
            key_features=key_features,
            primary_diagnosis=primary_diagnosis,
            differentials=differentials,
            distinguishing=distinguishing
        )
        
        messages = [
            {"role": "system", "content": "You are a medical information retrieval expert. Generate precise, targeted search queries."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse queries
        queries = []
        pattern = r'Query\s*\d+\s*:\s*(.+?)(?=Query\s*\d+\s*:|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            query = match.strip().strip('[]')
            if query and len(query) > 5:
                queries.append(query)
        
        # Fallback if parsing fails
        if len(queries) < 3:
            queries = [
                f"{primary_diagnosis} {key_features}",
                f"{primary_diagnosis} vs {differentials} differential diagnosis",
                f"{distinguishing} diagnostic criteria",
                f"{primary_diagnosis} pathophysiology mechanism",
                f"{primary_diagnosis} treatment management",
            ][:5]
        
        return queries[:5]
    
    # =========================================================================
    # Planning V3: Adaptive Query Generation (dynamic number of queries)
    # =========================================================================
    async def generate_plan_v3_async(
        self,
        question: str,
        options: Dict[str, str],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Generate adaptive planning with complexity assessment"""
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        prompt = PLANNING_V3_PROMPT.format(question=question, options=options_text)
        
        messages = [
            {"role": "system", "content": "You are an expert medical diagnostician. Be concise and precise."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse the planning response
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                plan = json.loads(response[start:end])
            else:
                plan = {}
        except:
            plan = {}
        
        # Ensure required fields exist with defaults
        plan.setdefault("question_type", "diagnostic")
        plan.setdefault("complexity", "moderate")
        plan.setdefault("key_clinical_clues", [])
        plan.setdefault("most_likely_answer", "")
        plan.setdefault("confidence", "medium")
        plan.setdefault("what_evidence_needed", [])
        plan.setdefault("differential_if_uncertain", [])
        
        # Fallback for key clues
        if not plan["key_clinical_clues"]:
            words = re.findall(r'\b[A-Za-z]{4,}\b', question)
            plan["key_clinical_clues"] = words[:4] if words else ["symptom"]
        
        return plan
    
    async def generate_queries_from_plan_v3_async(
        self,
        question: str,
        plan: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> List[str]:
        """Generate adaptive number of queries based on the plan"""
        
        question_type = plan.get("question_type", "diagnostic")
        complexity = plan.get("complexity", "moderate")
        key_clues = ", ".join(plan.get("key_clinical_clues", [])[:4])
        likely_answer = plan.get("most_likely_answer", "unknown")
        evidence_needed = ", ".join(plan.get("what_evidence_needed", [])[:3])
        differentials = plan.get("differential_if_uncertain", [])
        
        # Build differential section only if needed
        differential_section = ""
        if differentials and plan.get("confidence") == "low":
            differential_section = f"- Differentials to Consider: {', '.join(differentials[:3])}"
        
        prompt = PLANNING_V3_QUERY_PROMPT.format(
            question=question,
            question_type=question_type,
            complexity=complexity,
            key_clues=key_clues,
            likely_answer=likely_answer,
            evidence_needed=evidence_needed,
            differential_section=differential_section
        )
        
        messages = [
            {"role": "system", "content": "You are a medical search expert. Generate only essential, highly targeted queries. Quality over quantity."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse queries (flexible number)
        queries = []
        pattern = r'Query\s*\d+\s*:\s*(.+?)(?=Query\s*\d+\s*:|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            query = match.strip().strip('[]').strip()
            if query and len(query) > 5 and not query.startswith("..."):
                queries.append(query)
        
        # Ensure minimum queries based on complexity
        if len(queries) < 2:
            # Fallback: generate minimal essential queries
            queries = [
                f"{likely_answer} {key_clues}",
                f"{evidence_needed}",
            ]
        
        # Cap at 7 queries max
        return queries[:7]
    
    # =========================================================================
    # Planning V4: Answer-Focused with Evidence Verification
    # =========================================================================
    async def generate_plan_v4_async(
        self,
        question: str,
        options: Dict[str, str],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Generate answer-focused plan with evidence requirements"""
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        prompt = PLANNING_V4_PROMPT.format(question=question, options=options_text)
        
        messages = [
            {"role": "system", "content": "You are an expert medical diagnostician. Make your best diagnostic guess and identify what evidence would confirm it."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse the planning response
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                plan = json.loads(response[start:end])
            else:
                plan = {}
        except:
            plan = {}
        
        # Ensure required fields
        plan.setdefault("discriminating_features", [])
        plan.setdefault("best_guess", "")
        plan.setdefault("reasoning", "")
        plan.setdefault("confirming_evidence", [])
        plan.setdefault("alternative_if_wrong", "")
        
        # Fallback for features
        if not plan["discriminating_features"]:
            words = re.findall(r'\b[A-Za-z]{4,}\b', question)
            plan["discriminating_features"] = words[:3] if words else ["symptom"]
        
        return plan
    
    async def generate_queries_from_plan_v4_async(
        self,
        question: str,
        plan: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> List[str]:
        """Generate 3 targeted queries based on the answer-focused plan"""
        
        best_guess = plan.get("best_guess", "")
        reasoning = plan.get("reasoning", "")
        confirming_evidence = ", ".join(plan.get("confirming_evidence", [])[:3])
        discriminating_features = ", ".join(plan.get("discriminating_features", [])[:3])
        
        prompt = PLANNING_V4_QUERY_PROMPT.format(
            question=question,
            best_guess=best_guess,
            reasoning=reasoning,
            confirming_evidence=confirming_evidence,
            discriminating_features=discriminating_features
        )
        
        messages = [
            {"role": "system", "content": "Generate highly specific medical search queries."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse queries
        queries = []
        pattern = r'Query\s*\d+\s*:\s*(.+?)(?=Query\s*\d+\s*:|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            query = match.strip().strip('[]').strip()
            if query and len(query) > 5:
                queries.append(query)
        
        # Fallback if parsing fails
        if len(queries) < 2:
            queries = [
                f"{best_guess} {discriminating_features}",
                f"{confirming_evidence}",
                f"{reasoning} diagnosis",
            ]
        
        return queries[:5]  # Cap at 5
    
    # =========================================================================
    # Planning V4 + GRPO: Trained Rewriter Adapter
    # =========================================================================
    def generate_queries_grpo_rewriter(
        self,
        question: str,
        options: Dict[str, str],
        plan: Dict[str, Any],
    ) -> List[str]:
        """Generate queries using the GRPO-trained rewriter (HF model w/ LoRA).
        
        Synchronous method — runs the HF model with LoRA adapter.
        Call via asyncio.loop.run_in_executor() when inside an async context.
        """
        if self.grpo_rewriter_model is None:
            raise RuntimeError("GRPO rewriter not loaded. Use --rewriter-checkpoint.")
        
        import torch
        from data.medqa_loader import format_rewriter_prompt
        from training.reward import parse_queries_from_completion
        
        messages = format_rewriter_prompt(question, options, plan)
        try:
            prompt_text = self.grpo_rewriter_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = f"system: {messages[0]['content']}\nuser: {messages[1]['content']}\nassistant:"
        
        inputs = self.grpo_rewriter_tokenizer(
            prompt_text, return_tensors="pt"
        ).to(self.grpo_rewriter_model.device)
        
        with torch.no_grad():
            outputs = self.grpo_rewriter_model.generate(
                **inputs, max_new_tokens=256,
                do_sample=EVAL_FIXED_DO_SAMPLE,
            )
        
        completion_ids = outputs[0][inputs["input_ids"].shape[1]:]
        completion_text = self.grpo_rewriter_tokenizer.decode(
            completion_ids, skip_special_tokens=True
        )
        
        queries = parse_queries_from_completion(completion_text)
        
        # Fallback if parsing fails
        if len(queries) < 2:
            best_guess = plan.get("best_guess", "")
            confirming_evidence = ", ".join(plan.get("confirming_evidence", [])[:3])
            discriminating_features = ", ".join(plan.get("discriminating_features", [])[:3])
            queries = [
                f"{best_guess} {discriminating_features}",
                f"{confirming_evidence}",
                f"{plan.get('reasoning', '')} diagnosis",
            ]
        
        return queries[:5]
    
    # =========================================================================
    # Planning V6: Dual Hypothesis Testing
    # =========================================================================
    async def generate_plan_v6_async(
        self,
        question: str,
        options: Dict[str, str],
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Generate dual hypothesis plan for differential diagnosis"""
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        prompt = PLANNING_V6_PROMPT.format(question=question, options=options_text)
        
        messages = [
            {"role": "system", "content": "You are an expert medical diagnostician. Use differential diagnosis to generate two competing hypotheses."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse the planning response
        try:
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                plan = json.loads(response[start:end])
            else:
                plan = {}
        except:
            plan = {}
        
        # Ensure required fields
        if "hypothesis_1" not in plan:
            plan["hypothesis_1"] = {
                "diagnosis": "",
                "key_reasoning": "",
                "supporting_features": [],
                "evidence_to_verify": []
            }
        if "hypothesis_2" not in plan:
            plan["hypothesis_2"] = {
                "diagnosis": "",
                "key_reasoning": "",
                "supporting_features": [],
                "evidence_to_verify": []
            }
        if "discriminating_criteria" not in plan:
            plan["discriminating_criteria"] = []
        
        return plan
    
    async def generate_queries_from_plan_v6_async(
        self,
        question: str,
        plan: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> List[str]:
        """Generate 5-6 queries based on dual hypothesis plan"""
        
        h1 = plan.get("hypothesis_1", {})
        h2 = plan.get("hypothesis_2", {})
        
        h1_diagnosis = h1.get("diagnosis", "unknown")
        h1_reasoning = h1.get("key_reasoning", "")
        h1_evidence = ", ".join(h1.get("evidence_to_verify", [])[:2])
        
        h2_diagnosis = h2.get("diagnosis", "unknown")
        h2_reasoning = h2.get("key_reasoning", "")
        h2_evidence = ", ".join(h2.get("evidence_to_verify", [])[:2])
        
        discriminating = ", ".join(plan.get("discriminating_criteria", [])[:3])
        
        prompt = PLANNING_V6_QUERY_PROMPT.format(
            question=question,
            h1_diagnosis=h1_diagnosis,
            h1_reasoning=h1_reasoning,
            h1_evidence=h1_evidence,
            h2_diagnosis=h2_diagnosis,
            h2_reasoning=h2_reasoning,
            h2_evidence=h2_evidence,
            discriminating=discriminating
        )
        
        messages = [
            {"role": "system", "content": "Generate precise queries to test both hypotheses and distinguish between them."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        
        # Parse queries
        queries = []
        pattern = r'Query\s*\d+\s*(?:\(optional\))?\s*:\s*(.+?)(?=Query\s*\d+|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            query = match.strip().strip('[]').strip()
            if query and len(query) > 5 and not query.startswith("..."):
                queries.append(query)
        
        # Fallback if parsing fails
        if len(queries) < 3:
            queries = [
                f"{h1_diagnosis} {h1_evidence}",
                f"{h2_diagnosis} {h2_evidence}",
                f"{h1_diagnosis} vs {h2_diagnosis} differential",
                f"{discriminating} distinguishing features",
                f"clinical presentation comparison",
            ]
        
        return queries[:6]  # Cap at 6
    
    async def generate_answer_cot_async(
        self,
        question: str,
        options: Dict[str, str],
        session: aiohttp.ClientSession,
    ) -> Tuple[str, str]:
        """Generate answer using Chain-of-Thought (no RAG) asynchronously"""
        
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        user_prompt = COT_USER_PROMPT_TEMPLATE.format(
            question=question,
            options=options_text,
        )
        
        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = await self.client.chat_completion(messages, session=session)
        predicted = parse_answer(raw_response)
        return raw_response, predicted
    
    async def generate_answer_async(
        self,
        question: str,
        options: Dict[str, str],
        retrieved_docs: List[Dict[str, Any]],
        session: aiohttp.ClientSession,
    ) -> Tuple[str, str]:
        """Generate answer from evidence asynchronously"""
        
        context_parts = []
        for idx, doc in enumerate(retrieved_docs[:25]):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            context_parts.append(f"Document [{idx+1}] (Title: {title})\n{content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No documents."
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        user_prompt = MEDRAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            options=options_text,
        )
        
        messages = [
            {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = await self.client.chat_completion(messages, session=session)
        predicted = parse_answer(raw_response)
        return raw_response, predicted
    
    # =========================================================================
    # Planning V5: Include Plan and Subqueries in Final Generation
    # =========================================================================
    async def generate_answer_with_plan_async(
        self,
        question: str,
        options: Dict[str, str],
        retrieved_docs: List[Dict[str, Any]],
        plan: Dict[str, Any],
        subqueries: List[str],
        session: aiohttp.ClientSession,
    ) -> Tuple[str, str]:
        """Generate answer with plan information and subqueries provided to LLM"""
        
        # Format plan information
        plan_text = ""
        if plan:
            if plan.get("best_guess"):
                plan_text += f"Initial Hypothesis: {plan['best_guess']}\n"
            if plan.get("reasoning"):
                plan_text += f"Reasoning: {plan['reasoning']}\n"
            if plan.get("discriminating_features"):
                features = ", ".join(plan['discriminating_features'][:3])
                plan_text += f"Key Discriminating Features: {features}\n"
            if plan.get("confirming_evidence"):
                evidence = ", ".join(plan['confirming_evidence'][:3])
                plan_text += f"Evidence to Confirm: {evidence}\n"
        
        # Format subqueries
        queries_text = ""
        if subqueries:
            queries_text = "Search Queries Used:\n"
            for i, q in enumerate(subqueries, 1):
                queries_text += f"  {i}. {q}\n"
        
        # Format documents
        context_parts = []
        for idx, doc in enumerate(retrieved_docs[:25]):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            context_parts.append(f"Document [{idx+1}] (Title: {title})\n{content}")
        
        context = "\n\n".join(context_parts) if context_parts else "No documents."
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
        
        # Construct prompt with plan and queries
        user_prompt = f"""
Question:
{question}

Options:
{options_text}

Initial Analysis Plan:
{plan_text}

Generated subqueries from the initial analysis:
{queries_text}

Retrieved Evidence:
{context}

Based on the initial analysis and the retrieved evidence, please provide your final answer. Think step-by-step and generate your output in json:"""
        
        messages = [
            {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = await self.client.chat_completion(messages, session=session)
        predicted = parse_answer(raw_response)
        return raw_response, predicted
    
    async def evaluate_question_cot_only(
        self,
        question_data: Dict[str, Any],
        question_id: int,
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Evaluate a single question using CoT only (no RAG)"""
        
        question = question_data['question']
        options = question_data['options']
        correct_answer = question_data.get('answer_idx', question_data.get('answer', ''))
        
        result = {
            "question_id": question_id,
            "question": question,  # Full question
            "options": options,
            "correct_answer": correct_answer,
            "modes": {},
        }
        
        try:
            raw_response, predicted = await self.generate_answer_cot_async(question, options, session)
            
            result["modes"]["cot"] = {
                "num_queries": 0,
                "num_docs": 0,
                "queries": [],
                "retrieved_docs": [],
                "raw_response": raw_response,
                "predicted_answer": predicted,
                "is_correct": predicted.upper() == correct_answer.upper(),
            }
            
        except Exception as e:
            result["error"] = str(e)
            result["modes"]["cot"] = {"is_correct": False, "error": str(e)}
        
        return result
    
    async def evaluate_question_selected_modes(
        self,
        question_data: Dict[str, Any],
        question_id: int,
        session: aiohttp.ClientSession,
        modes: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate one question for baseline modes only:
        cot/directrag/directrewriting.

        `planning*` variants are intentionally excluded from evaluate.py baseline
        path to keep behavior aligned with evaluate_past usage.
        """

        if modes is None:
            modes = ["cot", "directrag", "directrewriting"]
        modes = normalize_mode_list(modes)

        allowed_modes = {"cot", "directrag", "directrewriting"}
        invalid_modes = [mode for mode in modes if mode not in allowed_modes]
        if invalid_modes:
            raise ValueError(
                f"Unsupported baseline mode(s): {invalid_modes}. "
                f"Allowed: {sorted(allowed_modes)}"
            )

        question = question_data['question']
        options = question_data['options']
        correct_answer = question_data.get('answer_idx', question_data.get('answer', ''))
        
        result = {
            "question_id": question_id,
            "question": question,  # Full question, not truncated
            "options": options,
            "correct_answer": correct_answer,
            "modes": {},
        }
        
        try:
            # Handle CoT mode (no RAG)
            if "cot" in modes:
                cot_raw, cot_pred = await self.generate_answer_cot_async(question, options, session)
                result["modes"]["cot"] = {
                    "num_queries": 0,
                    "num_docs": 0,
                    "queries": [],
                    "retrieved_docs": [],
                    "raw_response": cot_raw,
                    "predicted_answer": cot_pred,
                    "is_correct": cot_pred.upper() == correct_answer.upper(),
                }

            rag_modes = [mode for mode in modes if mode in ["directrag", "directrewriting"]]
            if rag_modes:
                query_tasks = []
                query_task_names = []
                if "directrewriting" in rag_modes:
                    query_tasks.append(
                        self.generate_subqueries_async(
                            question,
                            plan=None,
                            session=session,
                            system_prompt=DIRECT_REWRITING_SYSTEM_PROMPT,
                            target_num_queries=DIRECT_REWRITING_TARGET_QUERIES,
                        )
                    )
                    query_task_names.append("directrewriting_queries")

                query_outputs = await asyncio.gather(*query_tasks) if query_tasks else []
                directrewriting_queries: List[str] = []
                for i, name in enumerate(query_task_names):
                    if name == "directrewriting_queries":
                        directrewriting_queries = query_outputs[i]

                total_docs = self.total_docs
                directrag_docs: List[Dict[str, Any]] = []
                directrewriting_docs: List[Dict[str, Any]] = []
                if "directrag" in rag_modes:
                    directrag_docs = self.retrieve_documents(
                        [question],
                        total_docs,
                        max_docs=total_docs,
                    )
                if "directrewriting" in rag_modes and directrewriting_queries:
                    # Fixed setting: 3 subqueries × 5 docs per query = 15 docs.
                    k = DIRECT_REWRITING_DOCS_PER_QUERY
                    directrewriting_docs = self.retrieve_documents(
                        directrewriting_queries,
                        k,
                        max_docs=DIRECT_REWRITING_TOTAL_DOCS,
                    )

                answer_tasks = []
                answer_mode_order = []
                if "directrag" in rag_modes:
                    answer_tasks.append(
                        self.generate_answer_async(question, options, directrag_docs, session)
                    )
                    answer_mode_order.append("directrag")
                if "directrewriting" in rag_modes and directrewriting_queries:
                    answer_tasks.append(
                        self.generate_answer_async(question, options, directrewriting_docs, session)
                    )
                    answer_mode_order.append("directrewriting")

                answer_results = await asyncio.gather(*answer_tasks) if answer_tasks else []
                for i, mode in enumerate(answer_mode_order):
                    raw_response, predicted_answer = answer_results[i]
                    if mode == "directrag":
                        docs = directrag_docs
                        queries = [question]
                    else:
                        docs = directrewriting_docs
                        queries = directrewriting_queries

                    result["modes"][mode] = {
                        "num_queries": len(queries),
                        "num_docs": len(docs),
                        "queries": queries,
                        "plan": None,
                        "retrieved_docs": docs,
                        "raw_response": raw_response,
                        "predicted_answer": predicted_answer,
                        "is_correct": predicted_answer.upper() == correct_answer.upper(),
                    }
            
        except Exception as e:
            result["error"] = str(e)
            for mode in modes:
                if mode not in result["modes"]:
                    result["modes"][mode] = {"is_correct": False, "error": str(e)}
        
        return result
    
    async def evaluate_question_all_modes(
        self,
        question_data: Dict[str, Any],
        question_id: int,
        session: aiohttp.ClientSession,
    ) -> Dict[str, Any]:
        """Evaluate a single question across cot/directrag/directrewriting."""
        return await self.evaluate_question_selected_modes(
            question_data, question_id, session,
            modes=["cot", "directrag", "directrewriting"]
        )


async def run_evaluation_async(
    max_questions: int = 100,
    llm_provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    max_concurrent: int = 100,
    output_dir: Optional[str] = None,
    modes: List[str] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_tokens: int = 4096,
    vllm_max_concurrent: int = 1,
    vllm_max_model_len: int = 8192,
    question_timeout_seconds: int = 600,
    total_docs: int = 15,
    question_dataset: str = "medqa",
    rewriter_adapter_path: Optional[str] = None,
    rewriter_base_model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Run maximum performance async evaluation with selectable modes"""
    set_global_seed(EVAL_FIXED_SEED)
    if modes is None:
        modes = ["cot", "directrag", "directrewriting"]
    modes = normalize_mode_list(modes)

    allowed_modes = {"cot", "directrag", "directrewriting"}
    invalid_modes = [mode for mode in modes if mode not in allowed_modes]
    if invalid_modes:
        raise ValueError(
            f"Unsupported baseline mode(s): {invalid_modes}. "
            f"Allowed: {sorted(allowed_modes)}"
        )

    openai_api_key = None
    if llm_provider == "openai":
        openai_api_key = resolve_openai_api_key()
        if not openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set env var or export it in ~/.bashrc."
            )

    output_dir = resolve_results_output_dir(
        output_dir=output_dir,
        llm_provider=llm_provider,
        model_name=model_name,
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MedQA Evaluation (MAX ASYNC)")
    print("=" * 80)
    print(f"LLM provider: {llm_provider}")
    print(f"Model: {model_name}")
    print(f"Question dataset: {question_dataset}")
    print(f"Modes: {modes}")
    print(f"Max questions: {max_questions}")
    if llm_provider == "vllm":
        print("Max concurrent requests: n/a (phased local vLLM baseline)")
    else:
        print(f"Max concurrent requests: {max_concurrent}")
    print(f"Fixed seed: {EVAL_FIXED_SEED}")
    print(f"Fixed temperature: {EVAL_FIXED_TEMPERATURE}")
    print(f"Fixed do_sample: {EVAL_FIXED_DO_SAMPLE} (where supported)")
    if llm_provider == "vllm":
        print(f"vLLM settings: tp={vllm_tensor_parallel_size}, gpu_mem={vllm_gpu_memory_utilization}, max_model_len={vllm_max_model_len}, max_tokens={vllm_max_tokens}")
        print("vLLM baseline batching: directrag/directrewriting run as phase batches.")
    if llm_provider != "vllm":
        print(f"Question timeout: {question_timeout_seconds}s")
    
    if "cot" in modes:
        print(f"CoT: No retrieval (Chain-of-Thought only)")
    
    rag_modes = [m for m in modes if m != "cot"]
    if rag_modes:
        if "directrag" in rag_modes:
            print(f"directrag: total_docs={total_docs}")
        if "directrewriting" in rag_modes:
            print(
                "directrewriting: "
                f"{DIRECT_REWRITING_TARGET_QUERIES} queries x "
                f"{DIRECT_REWRITING_DOCS_PER_QUERY} docs = "
                f"{DIRECT_REWRITING_TOTAL_DOCS}"
            )
        print(f"Retriever: {retriever_name} | Corpus: {corpus_name}")
    
    print("=" * 80)
    
    # Load dataset
    dataset = MedQADataset(question_dataset=question_dataset)
    capped_total = max(0, min(max_questions, len(dataset)))
    question_indices = list(range(capped_total))
    total_questions = len(question_indices)
    
    # Initialize evaluator
    evaluator = AsyncRAGEvaluator(
        llm_provider=llm_provider,
        model_name=model_name,
        retriever_name=retriever_name,
        corpus_name=corpus_name,
        api_key=openai_api_key,
        max_concurrent=max_concurrent,
        total_docs=total_docs,
        rewriter_adapter_path=rewriter_adapter_path,
        rewriter_base_model=rewriter_base_model,
        vllm_tensor_parallel_size=vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_tokens=vllm_max_tokens,
        vllm_max_concurrent=vllm_max_concurrent,
        vllm_max_model_len=vllm_max_model_len,
        api_base=api_base,
    )
    
    if rag_modes:
        print(f"\nInitializing {corpus_name} retriever (this may take several minutes for large datasets like PubMed)...")
        if hasattr(evaluator.retriever, "_lazy_init"):
            evaluator.retriever._lazy_init()
            
    results = []
    mode_stats = {mode: {"correct": 0, "total": 0} for mode in modes}
    
    start_time = time.time()
    
    # Create aiohttp session
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def evaluate_with_timeout(question_data: Dict[str, Any], question_id: int) -> Dict[str, Any]:
            try:
                return await asyncio.wait_for(
                    evaluator.evaluate_question_selected_modes(question_data, question_id, session, modes=modes),
                    timeout=question_timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_text = f"Error: Question timeout after {question_timeout_seconds}s"
            except Exception as e:
                error_text = f"Error: {str(e)}"

            question = question_data.get("question", "")
            options = question_data.get("options", {})
            correct_answer = question_data.get("answer_idx", question_data.get("answer", ""))
            return {
                "question_id": question_id,
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "error": error_text,
                "modes": {
                    mode: {
                        "is_correct": False,
                        "error": error_text,
                    }
                    for mode in modes
                },
            }
        
        if llm_provider == "vllm":
            print(f"\nProcessing {total_questions} questions with phased vLLM batches...")
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            questions = [dataset[question_id] for question_id in question_indices]
            directrewriting_queries: List[List[str]] = [[] for _ in range(total_questions)]
            directrag_docs: List[List[Dict[str, Any]]] = [[] for _ in range(total_questions)]
            directrewriting_docs: List[List[Dict[str, Any]]] = [[] for _ in range(total_questions)]
            raw_outputs_by_mode = {mode: [""] * total_questions for mode in modes}

            batch_client = evaluator.client
            if not hasattr(batch_client, "chat_completion_batch"):
                raise RuntimeError("vLLM client does not support batch completions.")

            if "directrewriting" in modes and questions:
                query_messages = [
                    _build_directrewriting_query_messages(qd["question"])
                    for qd in questions
                ]
                query_texts = await batch_client.chat_completion_batch(
                    query_messages,
                    session=session,
                )
                for i, text in enumerate(query_texts):
                    parsed_queries = parse_subqueries(
                        text,
                        DIRECT_REWRITING_TARGET_QUERIES,
                    )
                    directrewriting_queries[i] = _expand_queries_to_target_count(
                        parsed_queries,
                        questions[i]["question"],
                        DIRECT_REWRITING_TARGET_QUERIES,
                    )

            if "directrag" in modes:
                for i, qd in enumerate(questions):
                    directrag_docs[i] = evaluator.retrieve_documents(
                        [qd["question"]],
                        total_docs,
                        max_docs=total_docs,
                    )

            if "directrewriting" in modes:
                for i, queries in enumerate(directrewriting_queries):
                    directrewriting_docs[i] = evaluator.retrieve_documents(
                        queries,
                        DIRECT_REWRITING_DOCS_PER_QUERY,
                        max_docs=DIRECT_REWRITING_TOTAL_DOCS,
                    )

            batched_jobs: List[Tuple[str, int, List[Dict[str, str]]]] = []
            if "cot" in modes:
                for i, qd in enumerate(questions):
                    batched_jobs.append(
                        ("cot", i, _build_cot_messages(qd["question"], qd["options"]))
                    )
            if "directrag" in modes:
                for i, qd in enumerate(questions):
                    batched_jobs.append(
                        (
                            "directrag",
                            i,
                            _build_medrag_answer_messages(
                                qd["question"],
                                qd["options"],
                                directrag_docs[i],
                            ),
                        )
                    )
            if "directrewriting" in modes:
                for i, qd in enumerate(questions):
                    batched_jobs.append(
                        (
                            "directrewriting",
                            i,
                            _build_medrag_answer_messages(
                                qd["question"],
                                qd["options"],
                                directrewriting_docs[i],
                            ),
                        )
                    )

            if batched_jobs:
                batch_texts = await batch_client.chat_completion_batch(
                    [job[2] for job in batched_jobs],
                    session=session,
                )
                for (mode, question_idx, _), raw_text in zip(batched_jobs, batch_texts):
                    raw_outputs_by_mode[mode][question_idx] = raw_text

            for i, qd in enumerate(questions):
                correct_answer = str(qd.get("answer_idx", qd.get("answer", "")) or "")
                gold = correct_answer.upper()
                result = {
                    "question_id": question_indices[i],
                    "question": qd["question"],
                    "options": qd["options"],
                    "correct_answer": correct_answer,
                    "modes": {},
                }

                if "cot" in modes:
                    raw_response = raw_outputs_by_mode["cot"][i]
                    predicted = parse_answer(raw_response)
                    is_correct = predicted.upper() == gold
                    result["modes"]["cot"] = {
                        "num_queries": 0,
                        "num_docs": 0,
                        "queries": [],
                        "retrieved_docs": [],
                        "raw_response": raw_response,
                        "predicted_answer": predicted,
                        "is_correct": is_correct,
                    }

                if "directrag" in modes:
                    raw_response = raw_outputs_by_mode["directrag"][i]
                    predicted = parse_answer(raw_response)
                    docs = directrag_docs[i]
                    is_correct = predicted.upper() == gold
                    result["modes"]["directrag"] = {
                        "num_queries": 1,
                        "num_docs": len(docs),
                        "queries": [qd["question"]],
                        "plan": None,
                        "retrieved_docs": _serialize_retrieved_docs_for_output(docs),
                        "raw_response": raw_response,
                        "predicted_answer": predicted,
                        "is_correct": is_correct,
                    }

                if "directrewriting" in modes:
                    raw_response = raw_outputs_by_mode["directrewriting"][i]
                    predicted = parse_answer(raw_response)
                    docs = directrewriting_docs[i]
                    queries = directrewriting_queries[i]
                    is_correct = predicted.upper() == gold
                    result["modes"]["directrewriting"] = {
                        "num_queries": len(queries),
                        "num_docs": len(docs),
                        "queries": queries,
                        "plan": None,
                        "retrieved_docs": _serialize_retrieved_docs_for_output(docs),
                        "raw_response": raw_response,
                        "predicted_answer": predicted,
                        "is_correct": is_correct,
                    }

                results.append(result)
                for mode in modes:
                    mode_result = result.get("modes", {}).get(mode, {})
                    if "error" not in mode_result and "is_correct" in mode_result:
                        mode_stats[mode]["total"] += 1
                        if mode_result.get("is_correct", False):
                            mode_stats[mode]["correct"] += 1
        else:
            # Estimate time based on mode complexity
            api_calls_per_question = len([m for m in modes if m == "cot"])  # 1 for CoT
            if "directrag" in modes:
                api_calls_per_question += 1
            if "directrewriting" in modes:
                api_calls_per_question += 2  # subquery gen + answer

            # Run bounded in-flight tasks (avoid creating all question coroutines at once).
            print(f"\nProcessing {total_questions} questions with {max_concurrent} concurrent requests...")
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Estimated API calls: ~{total_questions * api_calls_per_question}")
            print("=" * 80)

            async def run_one(question_id: int) -> Dict[str, Any]:
                question_data = dataset[question_id]
                return await evaluate_with_timeout(question_data, question_id)

            max_in_flight = max(1, int(max_concurrent))
            question_iter = iter(question_indices)
            in_flight: Dict[asyncio.Task, int] = {}

            for _ in range(min(max_in_flight, total_questions)):
                try:
                    qid = next(question_iter)
                except StopIteration:
                    break
                task = asyncio.create_task(run_one(qid))
                in_flight[task] = qid

            completed_count = 0
            while in_flight:
                done, _ = await asyncio.wait(
                    set(in_flight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    _qid = in_flight.pop(task, None)
                    try:
                        result = task.result()
                    except Exception as e:
                        result = {
                            "question_id": _qid if _qid is not None else -1,
                            "error": f"Error: {str(e)}",
                            "modes": {mode: {"is_correct": False, "error": str(e)} for mode in modes},
                        }

                    completed_count += 1
                    if result is not None:
                        results.append(result)
                        for mode in modes:
                            mode_result = result.get("modes", {}).get(mode, {})
                            if "error" not in mode_result and "is_correct" in mode_result:
                                mode_stats[mode]["total"] += 1
                                if mode_result.get("is_correct", False):
                                    mode_stats[mode]["correct"] += 1

                    if completed_count % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed if elapsed > 0 else 0.0
                        eta_seconds = (total_questions - completed_count) / rate if rate > 0 else 0.0
                        print(
                            f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                            f"Progress: {completed_count}/{total_questions} "
                            f"({completed_count/total_questions*100:.1f}%)"
                        )
                        print(
                            f"  Elapsed: {elapsed/60:.1f} min | "
                            f"Speed: {rate*60:.1f} q/min | ETA: {eta_seconds/60:.1f} min"
                        )
                        for mode in modes:
                            stats = mode_stats[mode]
                            if stats["total"] > 0:
                                acc = stats["correct"] / stats["total"] * 100
                                print(
                                    f"  {mode:12s}: {stats['correct']:3d}/{stats['total']:3d} "
                                    f"({acc:.1f}%)"
                                )
                        print("-" * 80)

                    try:
                        next_qid = next(question_iter)
                    except StopIteration:
                        next_qid = None
                    if next_qid is not None:
                        next_task = asyncio.create_task(run_one(next_qid))
                        in_flight[next_task] = next_qid
    
    total_time = time.time() - start_time
    
    # Build summary
    summary = {
        "config": {
            "llm_provider": llm_provider,
            "model_name": model_name,
            "question_dataset": question_dataset,
            "modes": modes,
            "retriever_name": retriever_name if rag_modes else None,
            "corpus_name": corpus_name if rag_modes else None,
            "retrieval_dataset": (
                CORPUS_TO_RETRIEVAL_DATASET.get(corpus_name.lower())
                if rag_modes and corpus_name
                else None
            ),
            "max_questions": max_questions,
            "total_evaluated": len(results),
            "max_concurrent": max_concurrent,
            "vllm_tensor_parallel_size": vllm_tensor_parallel_size if llm_provider == "vllm" else None,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization if llm_provider == "vllm" else None,
            "vllm_max_tokens": vllm_max_tokens if llm_provider == "vllm" else None,
            "vllm_max_concurrent": vllm_max_concurrent if llm_provider == "vllm" else None,
            "vllm_max_model_len": vllm_max_model_len if llm_provider == "vllm" else None,
            "question_timeout_seconds": (
                question_timeout_seconds if llm_provider != "vllm" else None
            ),
            "total_docs": total_docs,
        },
        "timing": {
            "total_seconds": total_time,
            "avg_per_question": total_time / len(results) if results else 0,
            "questions_per_minute": len(results) / (total_time / 60) if total_time > 0 else 0,
        },
        "mode_results": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    for mode in modes:
        stats = mode_stats[mode]
        accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        summary["mode_results"][mode] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": accuracy,
        }
    
    # Print results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total questions: {len(results)}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
    print(f"Speed: {summary['timing']['questions_per_minute']:.1f} questions/minute")
    print()
    
    print("Accuracy by Mode:")
    print("-" * 40)
    for mode in modes:
        stats = summary["mode_results"][mode]
        print(f"  {mode:12s}: {stats['correct']:4d}/{stats['total']:4d} ({stats['accuracy']:.1f}%)")
    
    # Show comparison if multiple modes
    if len(modes) > 1:
        print()
        base_mode = modes[0]
        base_acc = summary["mode_results"][base_mode]["accuracy"]
        print(f"Comparison vs {base_mode}:")
        print("-" * 40)
        for mode in modes[1:]:
            mode_acc = summary["mode_results"][mode]["accuracy"]
            diff = mode_acc - base_acc
            print(f"  {mode:12s}: {diff:+.1f}%")
    
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_".join(modes)
    model_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    if not model_suffix:
        model_suffix = "model"
    output_file = os.path.join(
        output_dir,
        f"{question_dataset}_{mode_suffix}_{model_suffix}_{timestamp}.json",
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return summary


# ============================================================================
# Batch-Phased Evaluation (for checkpoint evaluation)
# ============================================================================
# Process ALL questions per phase with full VRAM utilization.
# Only ONE model on GPU at a time → gpu_mem=0.9, full vLLM batching.
#
# Each vLLM phase runs in a SEPARATE subprocess so GPU memory is
# naturally freed when the subprocess exits. Data passes via temp files.
#
# Phase 1: Plan ALL questions    (base model)      [subprocess]
# Phase 2: Rewrite ALL queries   (checkpoint model) [subprocess]
# Phase 3: Retrieve ALL          (CPU, in-process)
# Phase 4: Answer ALL questions  (base model)      [subprocess]
# ============================================================================

import pickle
import tempfile
import multiprocessing as _mp


def _format_chat_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Format chat messages into a prompt for vLLM generate()."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)


def _extract_vllm_output_texts(outputs) -> List[str]:
    """
    Normalize vLLM outputs (chat() or generate()) into a list of strings.
    Supports a few return shapes across vLLM versions.
    """
    if outputs is None:
        return []

    if isinstance(outputs, tuple):
        outputs = list(outputs)
    elif not isinstance(outputs, list):
        outputs = [outputs]

    texts = []
    for out in outputs:
        text = ""
        if hasattr(out, "outputs"):
            try:
                first = out.outputs[0]
                if hasattr(first, "text"):
                    text = first.text
            except Exception:
                text = ""
        elif isinstance(out, dict):
            if "choices" in out and out["choices"]:
                choice = out["choices"][0]
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        text = message.get("content", "") or text
                    else:
                        text = choice.get("text", text)
                else:
                    text = getattr(choice, "text", text)
            elif "text" in out:
                text = out.get("text", "")
        elif isinstance(out, str):
            text = out

        texts.append("" if text is None else str(text))

    return texts


def _resolve_vllm_model_and_lora(model_path: str):
    """
    Resolve a checkpoint path for vLLM.

    Returns:
      model_for_vllm: base model path/name to load into LLM(...)
      lora_request:   LoRARequest or None
      load_msg:       human-readable message for logging
      extra_llm_kwargs: kwargs to pass into LLM(...)
    """
    # Default: treat as a normal HF model or full checkpoint.
    model_for_vllm = model_path
    lora_request = None
    load_msg = f"Loading model: {os.path.basename(model_path)}"
    extra_llm_kwargs = {}

    # LoRA adapter checkpoints typically contain adapter_config.json but no config.json.
    if os.path.isdir(model_path):
        adapter_cfg = os.path.join(model_path, "adapter_config.json")
        has_hf_cfg = os.path.exists(os.path.join(model_path, "config.json"))
        has_mistral_cfg = os.path.exists(os.path.join(model_path, "params.json"))

        if os.path.exists(adapter_cfg) and not (has_hf_cfg or has_mistral_cfg):
            try:
                from vllm.lora.request import LoRARequest

                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                base_model = cfg.get("base_model_name_or_path")
                lora_rank = int(cfg.get("r", 16) or 16)
                if not base_model:
                    raise ValueError(
                        f"adapter_config.json missing base_model_name_or_path: {adapter_cfg}"
                    )

                # vLLM loads base model once, then applies LoRA per request.
                model_for_vllm = base_model
                lora_request = LoRARequest(
                    lora_name=os.path.basename(os.path.normpath(model_path)),
                    lora_int_id=1,
                    lora_path=model_path,
                )
                extra_llm_kwargs["enable_lora"] = True
                extra_llm_kwargs["max_lora_rank"] = max(8, lora_rank)
                load_msg = (
                    f"Loading base model: {base_model} + LoRA: "
                    f"{os.path.basename(model_path)} (r={lora_rank})"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to configure LoRA checkpoint '{model_path}': {e}"
                ) from e

    return model_for_vllm, lora_request, load_msg, extra_llm_kwargs


def _run_vllm_batch_phase(model_path: str, messages_list_file: str,
                          output_file: str, gpu_mem: float,
                          max_model_len: int, max_tokens: int,
                          tensor_parallel_size: int = 1):
    """
    Subprocess entry: load vLLM model, run batch chat, save outputs, exit.
    All GPU memory is freed when this process exits.
    """
    from vllm import LLM, SamplingParams

    # Load messages
    with open(messages_list_file, "rb") as f:
        messages_list = pickle.load(f)

    model_for_vllm, lora_request, load_msg, extra_llm_kwargs = _resolve_vllm_model_and_lora(model_path)
    print(f"    [subprocess] {load_msg} ...")
    llm = LLM(
        model=model_for_vllm,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enforce_eager=True,
        **extra_llm_kwargs,
    )

    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
    print(f"    [subprocess] Running batch inference on {len(messages_list)} inputs ...")

    prompts = [_format_chat_messages(llm.get_tokenizer(), m) for m in messages_list]
    try:
        outputs = llm.chat(
            messages=messages_list,
            sampling_params=sampling,
            use_tqdm=True,
            lora_request=lora_request,
        )
    except Exception:
        outputs = llm.generate(
            prompts,
            sampling_params=sampling,
            lora_request=lora_request,
            use_tqdm=True,
        )

    # Extract text outputs
    result_texts = _extract_vllm_output_texts(outputs)
    if len(result_texts) < len(messages_list):
        result_texts.extend([""] * (len(messages_list) - len(result_texts)))
    if len(result_texts) != len(messages_list):
        raise RuntimeError(
            f"Unexpected vLLM output count: got {len(result_texts)} for {len(messages_list)} inputs"
        )

    with open(output_file, "wb") as f:
        pickle.dump(result_texts, f)

    print(f"    [subprocess] Done. Exiting to free GPU memory.")
    # Process exit frees all GPU memory automatically


def _run_phase_in_subprocess(model_path: str, messages_list: list,
                             gpu_mem: float, max_model_len: int,
                             max_tokens: int,
                             tensor_parallel_size: int = 1) -> list:
    """Run a vLLM batch in a subprocess and return result texts."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in:
        pickle.dump(messages_list, f_in)
        input_path = f_in.name

    output_path = input_path + ".out.pkl"

    try:
        p = _mp.Process(
            target=_run_vllm_batch_phase,
            args=(model_path, input_path, output_path,
                  gpu_mem, max_model_len, max_tokens, tensor_parallel_size),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"vLLM subprocess failed (exit={p.exitcode})")

        with open(output_path, "rb") as f:
            return pickle.load(f)
    finally:
        for fp in [input_path, output_path]:
            try:
                os.unlink(fp)
            except OSError:
                pass


def _format_list(val):
    """Convert list or string to comma-separated string."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val[:5])
    return str(val)


def _build_cot_messages(
    question: str,
    options: Dict[str, str],
) -> List[Dict[str, str]]:
    """Build baseline CoT messages."""
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    return [
        {"role": "system", "content": COT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COT_USER_PROMPT_TEMPLATE.format(
                question=question,
                options=options_text,
            ),
        },
    ]


def _build_directrewriting_query_messages(question: str) -> List[Dict[str, str]]:
    """Build baseline directrewriting query-generation messages."""
    return [
        {"role": "system", "content": DIRECT_REWRITING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DIRECT_REWRITING_PROMPT.format(query=question),
        },
    ]


def _build_medrag_answer_messages(
    question: str,
    options: Dict[str, str],
    retrieved_docs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build baseline RAG answer-generation messages."""
    context_parts = []
    for idx, doc in enumerate(retrieved_docs[:25]):
        title = doc.get("title", "Untitled")
        content = doc.get("content", "")
        context_parts.append(f"Document [{idx+1}] (Title: {title})\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "No documents."
    options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    return [
        {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": MEDRAG_USER_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
                options=options_text,
            ),
        },
    ]


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first top-level JSON object from free-form text."""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None

    # Fast path
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Bracket matching with quote/escape handling.
    start_idx = s.find("{")
    while start_idx != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(s)):
            ch = s[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start_idx : i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break
        start_idx = s.find("{", start_idx + 1)
    return None


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
    """Backfill closest-alternative fields for v8/v11 compatibility."""
    extra = _extract_first_json_object(raw_text) or {}

    # Best guess text backfill (supports hv8 direct JSON output).
    bg_letter = coerce_choice(plan.get("best_guess", "") or extra.get("best_guess", ""))
    plan["best_guess"] = bg_letter
    plan["best_guess_text"] = _resolve_choice_text(
        options,
        bg_letter,
        str(plan.get("best_guess_text", "") or extra.get("best_guess_text", "")).strip(),
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
    # Keep legacy key aligned for older prompts/paths.
    plan["alternative_if_wrong"] = closest_letter

    return plan


async def _run_openai_batch_phase(
    client: AsyncOpenAIClient,
    session: aiohttp.ClientSession,
    messages_list: List[List[Dict[str, str]]],
    desc: str,
) -> List[str]:
    """Run one phase using OpenAI async client, preserving input order."""
    if not messages_list:
        return []
    tasks = [
        client.chat_completion(messages=messages, session=session)
        for messages in messages_list
    ]
    return await tqdm_asyncio.gather(*tasks, desc=desc)


async def run_batch_phased_evaluation_openai(
    base_model: str,
    rewriter_checkpoint: str = None,
    hypothesis_checkpoint: str = None,
    generator_checkpoint: str = None,
    max_questions: int = 1273,
    total_docs: int = 15,
    question_dataset: str = "medqa",
    max_tokens: int = 2048,
    max_concurrent: int = 100,
    output_dir: Optional[str] = None,
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    hypothesis_prompt: str = "v1",
    rewriting_prompt: str = "v1",
    generator_prompt: str = "v1",
    api_base: Optional[str] = None,
    measure_best_guess_only: bool = False,
    use_gating: bool = False,
    gating_mode: str = "both",
    gating_policy: str = "conf3_skip",
    trust_threshold: int = 3,
    trust_evaluator_model: Optional[str] = None,
    trust_evaluator_prompt: str = "v1",
    hypothesis_finalizer_prompt: str = "v1",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Batch-phased hypothesis evaluation via OpenAI API (GPT-compatible).

    This mirrors run_batch_phased_evaluation() but uses async API calls
    instead of vLLM subprocess inference.
    """
    from training.reward import (
        parse_queries_from_completion, parse_hypothesis_plan, parse_trust_evaluation,
    )
    if seed != EVAL_FIXED_SEED:
        print(f"INFO: overriding requested seed={seed} -> fixed seed={EVAL_FIXED_SEED}")
    seed = EVAL_FIXED_SEED
    set_global_seed(seed)

    output_dir = resolve_results_output_dir(
        output_dir=output_dir,
        llm_provider="openai",
        model_name=base_model,
    )

    api_key = resolve_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Set env or add export to ~/.bashrc.")

    # Resolve per-module models
    hypothesis_model = hypothesis_checkpoint or base_model
    rewriter_model = rewriter_checkpoint or base_model
    generator_model = generator_checkpoint or base_model
    trust_eval_model = trust_evaluator_model or base_model

    h_prompt = HYPOTHESIS_PROMPTS[hypothesis_prompt]
    r_prompt = REWRITING_PROMPTS[rewriting_prompt]
    g_prompt = GENERATOR_PROMPTS[generator_prompt]
    te_prompt = TRUST_EVALUATOR_PROMPTS[trust_evaluator_prompt]
    hf_prompt = HYPOTHESIS_FINALIZER_PROMPTS[hypothesis_finalizer_prompt]
    combo_tag = f"h{hypothesis_prompt}_r{rewriting_prompt}_g{generator_prompt}"

    run_self_conf_gating = use_gating and gating_mode in {"self_confidence", "both"}
    run_trust_gating = use_gating and gating_mode in {"trust_eval", "both"}

    hypothesis_client = AsyncOpenAIClient(
        api_key=api_key,
        model=hypothesis_model,
        max_concurrent=max_concurrent,
        api_base=api_base,
    )
    rewriter_client = AsyncOpenAIClient(
        api_key=api_key,
        model=rewriter_model,
        max_concurrent=max_concurrent,
        api_base=api_base,
    )
    generator_client = AsyncOpenAIClient(
        api_key=api_key,
        model=generator_model,
        max_concurrent=max_concurrent,
        api_base=api_base,
    )
    trust_evaluator_client = None
    if run_trust_gating:
        trust_evaluator_client = AsyncOpenAIClient(
            api_key=api_key,
            model=trust_eval_model,
            max_concurrent=max_concurrent,
            api_base=api_base,
        )

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = MedQADataset(question_dataset=question_dataset)
    capped_total = max(0, min(max_questions, len(dataset)))
    question_ids = list(range(capped_total))
    questions = [dataset[i] for i in question_ids]
    n = len(questions)
    print(f"\n{'='*70}")
    print(f"Batch-Phased Evaluation (OpenAI): {n} questions")
    print(f"Question set: {question_dataset}")
    print(f"Combo:        {combo_tag}")
    print(f"  hypothesis: {h_prompt['description']}")
    print(f"  rewriting:  {r_prompt['description']}")
    print(f"  generator:  {g_prompt['description']}")
    print(f"Base model:   {base_model}")
    print(f"Hypothesis:   {hypothesis_model}"
          f"{' (checkpoint override)' if hypothesis_checkpoint else ''}")
    print(f"Rewriter:     {rewriter_model}"
          f"{' (checkpoint override)' if rewriter_checkpoint else ''}")
    print(f"Generator:    {generator_model}"
          f"{' (checkpoint override)' if generator_checkpoint else ''}")
    print(f"max_concurrent={max_concurrent}  max_tokens={max_tokens}")
    print(f"seed={seed}")
    print(f"temperature={EVAL_FIXED_TEMPERATURE}")
    print(f"do_sample={EVAL_FIXED_DO_SAMPLE} (where supported)")
    if measure_best_guess_only:
        print("best_guess_only=True (planner phase only)")
    if use_gating:
        print(f"use_gating=True  gating_mode={gating_mode}")
        if run_self_conf_gating:
            print(f"  self_confidence policy={gating_policy}")
        if run_trust_gating:
            print(
                f"  trust_eval threshold={trust_threshold} "
                f"model={trust_eval_model} prompt={trust_evaluator_prompt} "
                f"finalizer_prompt={hypothesis_finalizer_prompt}"
            )
    if api_base:
        print(f"api_base={api_base}")
    print(f"{'='*70}")

    valid_gating_modes = {"self_confidence", "trust_eval", "both"}
    if gating_mode not in valid_gating_modes:
        raise ValueError(
            f"Unknown gating mode '{gating_mode}'. "
            f"Choose from: {sorted(valid_gating_modes)}"
        )
    valid_gating_policies = {"conf3_skip", "conf23_skip", "conf3_only_retrieve"}
    if run_self_conf_gating and gating_policy not in valid_gating_policies:
        raise ValueError(
            f"Unknown gating policy '{gating_policy}'. "
            f"Choose from: {sorted(valid_gating_policies)}"
        )
    if trust_threshold not in (1, 2, 3):
        raise ValueError("--trust-threshold must be one of {1,2,3}.")

    t_total = time.time()

    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        limit_per_host=max_concurrent,
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        # ── Phase 1: Hypothesis ALL (OpenAI async) ──
        print(f"\n[Phase 1/4] Generating hypotheses for {n} questions "
              f"({hypothesis_model}) ...")
        t1 = time.time()

        plan_messages = []
        for qd in questions:
            options_text = "\n".join(
                [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
            )
            user = h_prompt["user"].format(
                question=qd["question"],
                options=options_text,
            )
            plan_messages.append([
                {"role": "system", "content": h_prompt["system"]},
                {"role": "user", "content": user},
            ])

        plan_texts = await _run_openai_batch_phase(
            hypothesis_client,
            session,
            plan_messages,
            desc="Phase1-Hypothesis",
        )

        plans = []
        for idx, text in enumerate(plan_texts):
            p = parse_hypothesis_plan(text)
            p.setdefault("discriminating_features", [])
            p.setdefault("best_guess", "")
            p.setdefault("best_guess_text", "")
            p.setdefault("reasoning", "")
            p.setdefault("confirming_evidence", [])
            p.setdefault("alternative_if_wrong", "")
            p.setdefault("closest_alternative", "")
            p.setdefault("closest_alternative_text", "")
            p["confidence_level"] = coerce_confidence_level(
                p.get("confidence_level", 2),
                default=2,
            )
            if not p["discriminating_features"]:
                words = re.findall(r'\b[A-Za-z]{4,}\b', questions[idx]["question"])
                p["discriminating_features"] = words[:3] or ["symptom"]
            p = _enrich_plan_with_alternatives(
                p,
                text,
                questions[idx].get("options", {}),
            )
            plans.append(p)

        print(f"  ✓ Phase 1 done: {len(plans)} hypotheses "
              f"in {time.time()-t1:.1f}s")

        # ── Planner diagnostics: best_guess accuracy + confidence distribution ──
        best_guess_correct = 0
        closest_alternative_correct = 0
        closest_alternative_rescue = 0
        confidence_counts = {1: 0, 2: 0, 3: 0}
        planner_rows = []
        for i, qd in enumerate(questions):
            gold = coerce_choice(qd.get("answer_idx", qd.get("answer", "")))
            best_guess = coerce_choice(plans[i].get("best_guess", ""))
            closest_alternative = coerce_choice(
                plans[i].get("closest_alternative", plans[i].get("alternative_if_wrong", ""))
            )
            confidence = coerce_confidence_level(plans[i].get("confidence_level", 2), default=2)
            confidence_counts[confidence] += 1
            best_guess_is_correct = (best_guess == gold and bool(gold))
            closest_alt_is_correct = (closest_alternative == gold and bool(gold))
            if best_guess_is_correct:
                best_guess_correct += 1
            if closest_alt_is_correct:
                closest_alternative_correct += 1
                if not best_guess_is_correct:
                    closest_alternative_rescue += 1
            planner_rows.append({
                "question_id": question_ids[i],
                "gold": gold,
                "best_guess": best_guess,
                "best_guess_text": plans[i].get("best_guess_text", ""),
                "closest_alternative": closest_alternative,
                "closest_alternative_text": plans[i].get("closest_alternative_text", ""),
                "confidence_level": confidence,
                "best_guess_correct": best_guess_is_correct,
                "closest_alternative_correct": closest_alt_is_correct,
                "top2_hit": (best_guess_is_correct or closest_alt_is_correct),
            })

        best_guess_accuracy = (best_guess_correct / n * 100) if n > 0 else 0.0
        closest_alternative_accuracy = (closest_alternative_correct / n * 100) if n > 0 else 0.0
        top2_accuracy = (
            (best_guess_correct + closest_alternative_rescue) / n * 100 if n > 0 else 0.0
        )
        print(
            f"  ✓ best_guess accuracy: {best_guess_accuracy:.2f}% "
            f"({best_guess_correct}/{n})"
        )
        print(
            f"  ✓ closest_alternative accuracy: {closest_alternative_accuracy:.2f}% "
            f"({closest_alternative_correct}/{n}) | "
            f"top2={top2_accuracy:.2f}% (rescue={closest_alternative_rescue})"
        )
        print(
            "  ✓ confidence distribution: "
            f"L1={confidence_counts[1]} ({confidence_counts[1]/n*100:.1f}%), "
            f"L2={confidence_counts[2]} ({confidence_counts[2]/n*100:.1f}%), "
            f"L3={confidence_counts[3]} ({confidence_counts[3]/n*100:.1f}%)"
        )

        trust_levels = [2] * n
        trust_reasons = [""] * n
        trust_flags = [[] for _ in range(n)]
        trust_counts = {1: 0, 2: 0, 3: 0}
        if run_trust_gating:
            print(
                f"\n[Phase 1.5/4] Evaluating hypothesis trust for {n} questions "
                f"({trust_eval_model}) ..."
            )
            t15 = time.time()
            trust_messages = []
            for i, qd in enumerate(questions):
                options_text = "\n".join(
                    [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
                )
                hypothesis_json = json.dumps(plans[i], ensure_ascii=False)
                trust_messages.append([
                    {"role": "system", "content": te_prompt["system"]},
                    {
                        "role": "user",
                        "content": te_prompt["user"].format(
                            question=qd["question"],
                            options=options_text,
                            hypothesis_json=hypothesis_json,
                        ),
                    },
                ])

            trust_texts = await _run_openai_batch_phase(
                trust_evaluator_client,  # type: ignore[arg-type]
                session,
                trust_messages,
                desc="Phase1.5-TrustEval",
            )

            for i, text in enumerate(trust_texts):
                parsed = parse_trust_evaluation(text)
                level = coerce_confidence_level(
                    parsed.get("hypothesis_trust", 2),
                    default=2,
                )
                trust_levels[i] = level
                trust_reasons[i] = str(parsed.get("trust_reason", "") or "")
                trust_flags[i] = parsed.get("risk_flags", []) or []
                trust_counts[level] += 1

            print(
                "  ✓ trust distribution: "
                f"L1={trust_counts[1]} ({trust_counts[1]/n*100:.1f}%), "
                f"L2={trust_counts[2]} ({trust_counts[2]/n*100:.1f}%), "
                f"L3={trust_counts[3]} ({trust_counts[3]/n*100:.1f}%)"
            )
            print(f"  ✓ Phase 1.5 done in {time.time()-t15:.1f}s")

        if measure_best_guess_only:
            elapsed = time.time() - t_total
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", base_model).strip("_") or "model"
            mode_tag = f"bestguess_h{hypothesis_prompt}"
            out_json = os.path.join(
                output_dir,
                f"{question_dataset}_{mode_tag}_{model_suffix}_{ts}.json",
            )
            out_jsonl = os.path.join(
                output_dir,
                f"{question_dataset}_{mode_tag}_{model_suffix}_{ts}.jsonl",
            )
            out_csv = os.path.join(
                output_dir,
                f"{question_dataset}_{mode_tag}_{model_suffix}_{ts}.csv",
            )

            summary = {
                "config": {
                    "llm_provider": "openai",
                    "base_model": base_model,
                    "hypothesis_prompt": hypothesis_prompt,
                    "question_dataset": question_dataset,
                    "total_evaluated": n,
                    "measure_best_guess_only": True,
                    "seed": seed,
                    "command": " ".join(sys.argv),
                    "api_base": api_base,
                },
                "timing": {
                    "total_seconds": elapsed,
                    "avg_per_question": elapsed / n if n else 0,
                },
                "planner_results": {
                    "best_guess_accuracy": best_guess_accuracy,
                    "best_guess_correct": best_guess_correct,
                    "best_guess_total": n,
                    "closest_alternative_accuracy": closest_alternative_accuracy,
                    "closest_alternative_correct": closest_alternative_correct,
                    "closest_alternative_rescue": closest_alternative_rescue,
                    "top2_accuracy": top2_accuracy,
                    "confidence_counts": confidence_counts,
                    "confidence_ratios": {
                        "1": confidence_counts[1] / n if n else 0.0,
                        "2": confidence_counts[2] / n if n else 0.0,
                        "3": confidence_counts[3] / n if n else 0.0,
                    },
                    "trust_counts": trust_counts,
                    "trust_ratios": {
                        "1": trust_counts[1] / n if n else 0.0,
                        "2": trust_counts[2] / n if n else 0.0,
                        "3": trust_counts[3] / n if n else 0.0,
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }

            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": planner_rows}, f, indent=2, ensure_ascii=False)
            _write_jsonl(out_jsonl, planner_rows)
            _write_summary_csv(out_csv, {
                "mode": mode_tag,
                "model": base_model,
                "hypothesis_prompt": hypothesis_prompt,
                "best_guess_accuracy": round(best_guess_accuracy, 4),
                "best_guess_correct": best_guess_correct,
                "best_guess_total": n,
                "closest_alternative_accuracy": round(closest_alternative_accuracy, 4),
                "closest_alternative_correct": closest_alternative_correct,
                "closest_alternative_rescue": closest_alternative_rescue,
                "top2_accuracy": round(top2_accuracy, 4),
                "conf_1_ratio": round(confidence_counts[1] / n if n else 0.0, 6),
                "conf_2_ratio": round(confidence_counts[2] / n if n else 0.0, 6),
                "conf_3_ratio": round(confidence_counts[3] / n if n else 0.0, 6),
                "trust_1_ratio": round(trust_counts[1] / n if n else 0.0, 6),
                "trust_2_ratio": round(trust_counts[2] / n if n else 0.0, 6),
                "trust_3_ratio": round(trust_counts[3] / n if n else 0.0, 6),
                "seed": seed,
                "command": " ".join(sys.argv),
            })
            print(f"✓ Saved planner summary: {out_json}")
            print(f"✓ Saved planner details jsonl: {out_jsonl}")
            print(f"✓ Saved planner summary csv: {out_csv}")
            return {
                "combo": mode_tag,
                "accuracy": best_guess_accuracy,
                "correct": best_guess_correct,
                "total": n,
                "file": out_json,
            }

        # ── Phase 2: Rewrite ALL (OpenAI async) ──
        print(f"\n[Phase 2/4] Generating queries for {n} questions "
              f"({rewriter_model}) ...")
        t2 = time.time()

        rw_messages = []
        for i, qd in enumerate(questions):
            p = plans[i]
            options_text = "\n".join(
                [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
            )
            bg_letter = (p.get("best_guess", "") or "").strip().upper().rstrip(".")
            bg_text = p.get("best_guess_text") or (
                f"{bg_letter}. {qd['options'].get(bg_letter, bg_letter)}"
                if bg_letter in qd.get("options", {}) else bg_letter
            )
            alt_letter = (
                p.get("closest_alternative", "")
                or p.get("alternative_if_wrong", "")
                or ""
            ).strip().upper().rstrip(".")
            closest_alternative_text = p.get("closest_alternative_text") or ""
            alt_text = closest_alternative_text or (
                f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
                if alt_letter in qd.get("options", {})
                else alt_letter
            )
            user = r_prompt["user"].format(
                question=qd["question"],
                options=options_text,
                best_guess=p.get("best_guess", ""),
                best_guess_text=bg_text,
                reasoning=p.get("reasoning", ""),
                confirming_evidence=_format_list(p.get("confirming_evidence", [])),
                discriminating_features=_format_list(p.get("discriminating_features", [])),
                alternative_if_wrong=p.get("alternative_if_wrong", ""),
                alternative_text=alt_text,
                closest_alternative=p.get("closest_alternative", ""),
                closest_alternative_text=alt_text,
            )
            rw_messages.append([
                {"role": "system", "content": r_prompt["system"]},
                {"role": "user", "content": user},
            ])

        rw_texts = await _run_openai_batch_phase(
            rewriter_client,
            session,
            rw_messages,
            desc="Phase2-Rewriter",
        )

        all_queries = []
        for i, text in enumerate(rw_texts):
            queries = parse_queries_from_completion(text)
            if len(queries) < 2:
                p = plans[i]
                bg = p.get("best_guess", "")
                cf = _format_list(p.get("confirming_evidence", []))
                df = _format_list(p.get("discriminating_features", []))
                queries = [f"{bg} {df}", cf, f"{p.get('reasoning', '')} diagnosis"]
            all_queries.append(queries[:5])

        print(f"  ✓ Phase 2 done: {len(all_queries)} query sets "
              f"in {time.time()-t2:.1f}s")

        # ── Phase 3: Retrieve ALL (CPU, in-process) ──
        print(f"\n[Phase 3/4] Retrieving documents for {n} questions ...")
        t3 = time.time()

        try:
            from retrieval.retriever import create_retriever as _cr
            retriever = _cr(
                retriever_type="mirage",
                retriever_name=retriever_name,
                corpus_name=corpus_name,
            )
            if hasattr(retriever, "_lazy_init"):
                retriever._lazy_init()
        except Exception as e:
            print(f"  WARNING: Retriever init failed: {e}")
            retriever = None

        all_docs = []
        for i, queries in enumerate(all_queries):
            doc_scores: Dict[str, float] = {}
            doc_data: Dict[str, Dict[str, Any]] = {}
            if retriever:
                k_per = max(1, total_docs // max(len(queries), 1))
                for q in queries:
                    try:
                        docs, scores = retriever.retrieve(q, k=k_per)
                        for doc, score in zip(docs, scores):
                            doc_id = doc.get(
                                "id",
                                doc.get("title", str(hash(doc.get("content", "")[:100]))),
                            )
                            if doc_id not in doc_scores:
                                doc_scores[doc_id] = 0.0
                                doc_data[doc_id] = doc.copy()
                                doc_data[doc_id]["query_trace"] = []
                            try:
                                doc_scores[doc_id] += float(score)
                            except Exception:
                                doc_scores[doc_id] += 0.0
                            doc_data[doc_id]["query_trace"].append(q)
                    except Exception:
                        pass

            docs_sorted = []
            for doc_id in sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True):
                doc = doc_data[doc_id]
                doc["fused_score"] = doc_scores[doc_id]
                docs_sorted.append(doc)

            all_docs.append(docs_sorted)
            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{n} retrieved")

        simple_rag_docs = None
        if use_gating:
            print(f"\n[Phase 3b/4] Retrieving SIMPLE-RAG documents for {n} questions ...")
            t3b = time.time()
            simple_rag_docs = []
            for i, qd in enumerate(questions):
                simple_rag_docs.append(
                    _retrieve_fused_docs_for_query_set(
                        retriever,
                        [qd["question"]],
                        total_docs,
                    )
                )
                if (i + 1) % 200 == 0:
                    print(f"  ... {i+1}/{n} simple-rag retrieved")
            print(f"  ✓ Phase 3b done: simple-rag retrieval in {time.time()-t3b:.1f}s")

        print(f"  ✓ Phase 3 done: retrieval in {time.time()-t3:.1f}s")

        # Free retriever memory before phase 4
        if retriever is not None:
            del retriever
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("  ✓ Retriever memory freed")

        # ── Phase 4: Answer ALL (OpenAI async) ──
        print(f"\n[Phase 4/4] Generating answers for {n} questions "
              f"({generator_model}) ...")
        t4 = time.time()

        ans_messages = []
        for i, qd in enumerate(questions):
            options_text = "\n".join(
                [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
            )
            docs = all_docs[i]
            ctx = "\n\n".join(
                [
                    f"Document [{j+1}] (Title: {d.get('title', 'Untitled')})\n"
                    f"{d.get('content', '')}"
                    for j, d in enumerate(docs[:25])
                ]
            ) or "No documents."

            fmt_vars = {
                "context": ctx,
                "question": qd["question"],
                "options": options_text,
                "hypothesis_summary": "",
                "queries_summary": "",
                "best_guess": "",
                "best_guess_text": "",
                "closest_alternative": "",
                "closest_alternative_text": "",
            }
            if generator_prompt in {"v2", "v3"}:
                p = plans[i]
                bg_letter = (p.get("best_guess", "") or "").strip().upper().rstrip(".")
                bg_text = p.get("best_guess_text") or (
                    f"{bg_letter}. {qd['options'].get(bg_letter, bg_letter)}"
                    if bg_letter in qd.get("options", {})
                    else bg_letter
                )
                alt_letter = (
                    p.get("closest_alternative", "")
                    or p.get("alternative_if_wrong", "")
                    or ""
                ).strip().upper().rstrip(".")
                closest_alternative_text = p.get("closest_alternative_text") or ""
                alt_text = closest_alternative_text or (
                    f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
                    if alt_letter in qd.get("options", {})
                    else alt_letter
                )
                fmt_vars["hypothesis_summary"] = (
                    f"Best guess: {p.get('best_guess', '')} — {p.get('reasoning', '')}"
                )
                fmt_vars["queries_summary"] = "\n".join(
                    f"  {j+1}. {q}" for j, q in enumerate(all_queries[i])
                )
                fmt_vars["best_guess"] = p.get("best_guess", "")
                fmt_vars["best_guess_text"] = bg_text
                fmt_vars["closest_alternative"] = p.get("closest_alternative", "")
                fmt_vars["closest_alternative_text"] = alt_text

            user = g_prompt["user"].format(**fmt_vars)
            ans_messages.append([
                {"role": "system", "content": g_prompt["system"]},
                {"role": "user", "content": user},
            ])

        ans_texts = await _run_openai_batch_phase(
            generator_client,
            session,
            ans_messages,
            desc="Phase4-Generator",
        )

        no_rag_texts = None
        simple_rag_texts = None
        trust_finalizer_text_by_idx: Dict[int, str] = {}
        if use_gating:
            print(f"\n[Phase 4b/4] Generating No-RAG answers for {n} questions ...")
            t4b = time.time()
            no_rag_messages = []
            for qd in questions:
                options_text = "\n".join(
                    [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
                )
                no_rag_messages.append([
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": COT_USER_PROMPT_TEMPLATE.format(
                            question=qd["question"],
                            options=options_text,
                        ),
                    },
                ])
            no_rag_texts = await _run_openai_batch_phase(
                generator_client,
                session,
                no_rag_messages,
                desc="Phase4b-NoRAG",
            )
            print(f"  ✓ Phase 4b done: no-rag answers in {time.time()-t4b:.1f}s")

            print(f"\n[Phase 4c/4] Generating SIMPLE-RAG answers for {n} questions ...")
            t4c = time.time()
            simple_messages = []
            for i, qd in enumerate(questions):
                options_text = "\n".join(
                    [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
                )
                docs = (simple_rag_docs or [])[i] if simple_rag_docs is not None else []
                ctx = "\n\n".join(
                    [
                        f"Document [{j+1}] (Title: {d.get('title', 'Untitled')})\n"
                        f"{d.get('content', '')}"
                        for j, d in enumerate(docs[:25])
                    ]
                ) or "No documents."
                fmt_vars = {
                    "context": ctx,
                    "question": qd["question"],
                    "options": options_text,
                    "hypothesis_summary": "",
                    "queries_summary": "",
                    "best_guess": "",
                    "best_guess_text": "",
                    "closest_alternative": "",
                    "closest_alternative_text": "",
                }
                simple_user = g_prompt["user"].format(**fmt_vars)
                simple_messages.append([
                    {"role": "system", "content": g_prompt["system"]},
                    {"role": "user", "content": simple_user},
                ])
            simple_rag_texts = await _run_openai_batch_phase(
                generator_client,
                session,
                simple_messages,
                desc="Phase4c-SimpleRAG",
            )
            print(f"  ✓ Phase 4c done: simple-rag answers in {time.time()-t4c:.1f}s")

            if run_trust_gating:
                high_trust_indices = [
                    i for i, level in enumerate(trust_levels)
                    if is_high_trust(level, trust_threshold)
                ]
                if high_trust_indices:
                    print(
                        f"\n[Phase 4d/4] Generating hypothesis-finalizer answers for "
                        f"{len(high_trust_indices)} high-trust questions ..."
                    )
                    t4d = time.time()
                    finalizer_messages = []
                    for i in high_trust_indices:
                        qd = questions[i]
                        options_text = "\n".join(
                            [f"{k}. {v}" for k, v in sorted(qd["options"].items())]
                        )
                        finalizer_messages.append([
                            {"role": "system", "content": hf_prompt["system"]},
                            {
                                "role": "user",
                                "content": hf_prompt["user"].format(
                                    question=qd["question"],
                                    options=options_text,
                                    hypothesis_json=json.dumps(plans[i], ensure_ascii=False),
                                ),
                            },
                        ])

                    finalizer_texts = await _run_openai_batch_phase(
                        generator_client,
                        session,
                        finalizer_messages,
                        desc="Phase4d-HypothesisFinalizer",
                    )
                    for idx, text in zip(high_trust_indices, finalizer_texts):
                        trust_finalizer_text_by_idx[idx] = text
                    print(
                        f"  ✓ Phase 4d done: hypothesis-finalizer answers "
                        f"in {time.time()-t4d:.1f}s"
                    )
                else:
                    print("\n[Phase 4d/4] No high-trust samples, skipped finalizer phase.")

        print(f"  ✓ Phase 4 done: answers in {time.time()-t4:.1f}s")

    # ── Compile Results ──
    print(f"\n[Result] Compiling ...")
    results = []
    detail_rows = []
    hcqr_correct = 0
    no_rag_correct = 0
    simple_rag_correct = 0
    self_conf_gating_correct = 0
    trust_gating_correct = 0
    closest_alternative_correct = 0
    closest_alternative_rescue = 0
    total = n
    self_conf_retrieval_used_count = 0
    trust_retrieval_used_count = 0

    for i, qd in enumerate(questions):
        correct_answer = qd.get("answer_idx", qd.get("answer", ""))
        gold = coerce_choice(correct_answer)
        plan = plans[i]
        confidence_level = coerce_confidence_level(plan.get("confidence_level", 2), default=2)
        trust_level = coerce_confidence_level(
            trust_levels[i] if i < len(trust_levels) else 2,
            default=2,
        )
        trust_high = is_high_trust(trust_level, trust_threshold)
        best_guess = coerce_choice(plan.get("best_guess", ""))
        best_guess_is_correct = (best_guess == gold and bool(gold))
        closest_alternative = coerce_choice(
            plan.get("closest_alternative", plan.get("alternative_if_wrong", ""))
        )
        closest_alternative_is_correct = (closest_alternative == gold and bool(gold))
        if closest_alternative_is_correct:
            closest_alternative_correct += 1
            if not best_guess_is_correct:
                closest_alternative_rescue += 1

        raw_resp = ans_texts[i]
        predicted_hcqr = parse_answer(raw_resp)
        hcqr_is_correct = predicted_hcqr.upper() == gold.upper()
        if hcqr_is_correct:
            hcqr_correct += 1

        predicted_no_rag = ""
        no_rag_is_correct = False
        predicted_simple_rag = ""
        simple_rag_is_correct = False
        retrieve_for_self_conf = True
        pred_self_conf_gate = predicted_hcqr
        self_conf_gate_correct = hcqr_is_correct
        retrieve_for_trust = True
        pred_trust_gate = predicted_hcqr
        trust_gate_correct = hcqr_is_correct

        if use_gating:
            predicted_no_rag = parse_answer((no_rag_texts or [""])[i])
            no_rag_is_correct = predicted_no_rag.upper() == gold.upper()
            if no_rag_is_correct:
                no_rag_correct += 1

            predicted_simple_rag = parse_answer((simple_rag_texts or [""])[i])
            simple_rag_is_correct = predicted_simple_rag.upper() == gold.upper()
            if simple_rag_is_correct:
                simple_rag_correct += 1

            if run_self_conf_gating:
                retrieve_for_self_conf = should_use_retrieval(confidence_level, gating_policy)
                if retrieve_for_self_conf:
                    self_conf_retrieval_used_count += 1
                    pred_self_conf_gate = predicted_hcqr
                    self_conf_gate_correct = hcqr_is_correct
                else:
                    pred_self_conf_gate = predicted_no_rag
                    self_conf_gate_correct = no_rag_is_correct
                if self_conf_gate_correct:
                    self_conf_gating_correct += 1

            if run_trust_gating:
                retrieve_for_trust = not trust_high
                if retrieve_for_trust:
                    trust_retrieval_used_count += 1
                    pred_trust_gate = predicted_hcqr
                    trust_gate_correct = hcqr_is_correct
                else:
                    pred_trust_gate = parse_answer(trust_finalizer_text_by_idx.get(i, ""))
                    trust_gate_correct = pred_trust_gate.upper() == gold.upper()
                if trust_gate_correct:
                    trust_gating_correct += 1

        hcqr_docs_for_output = _serialize_retrieved_docs_for_output(all_docs[i])
        results.append({
            "question_id": question_ids[i],
            "question": qd["question"],
            "options": qd["options"],
            "correct_answer": gold,
            "modes": {
                combo_tag: {
                    "num_queries": len(all_queries[i]),
                    "num_docs": len(all_docs[i]),
                    "queries": all_queries[i],
                    "plan": plan,
                    "retrieved_docs": hcqr_docs_for_output,
                    "raw_response": raw_resp,
                    "predicted_answer": predicted_hcqr,
                    "is_correct": hcqr_is_correct,
                }
            },
            "planner": {
                "best_guess": best_guess,
                "best_guess_text": plan.get("best_guess_text", ""),
                "closest_alternative": closest_alternative,
                "closest_alternative_text": plan.get("closest_alternative_text", ""),
                "best_guess_is_correct": best_guess_is_correct,
                "closest_alternative_is_correct": closest_alternative_is_correct,
                "top2_hit": (best_guess_is_correct or closest_alternative_is_correct),
                "confidence_level": confidence_level,
                "trust_level": trust_level if run_trust_gating else None,
                "trust_reason": trust_reasons[i] if run_trust_gating else "",
                "risk_flags": trust_flags[i] if run_trust_gating else [],
            }
        })
        if use_gating:
            results[-1]["modes"]["no_rag"] = {
                "predicted_answer": predicted_no_rag,
                "is_correct": no_rag_is_correct,
            }
            results[-1]["modes"]["simple_rag"] = {
                "num_queries": 1,
                "num_docs": len((simple_rag_docs or [])[i]) if simple_rag_docs is not None else 0,
                "queries": [qd["question"]],
                "retrieved_docs": _serialize_retrieved_docs_for_output(
                    (simple_rag_docs or [])[i] if simple_rag_docs is not None else []
                ),
                "raw_response": (simple_rag_texts or [""])[i] if simple_rag_texts is not None else "",
                "predicted_answer": predicted_simple_rag,
                "is_correct": simple_rag_is_correct,
            }
            if run_self_conf_gating:
                results[-1]["modes"]["self_confidence_gating"] = {
                    "policy": gating_policy,
                    "retrieval_used": retrieve_for_self_conf,
                    "predicted_answer": pred_self_conf_gate,
                    "is_correct": self_conf_gate_correct,
                }
            if run_trust_gating:
                results[-1]["modes"]["trust_gating"] = {
                    "threshold": trust_threshold,
                    "trust_level": trust_level,
                    "trust_high": trust_high,
                    "retrieval_used": retrieve_for_trust,
                    "predicted_answer": pred_trust_gate,
                    "is_correct": trust_gate_correct,
                }
        detail_rows.append({
            "question_id": question_ids[i],
            "gold": gold,
            "best_guess": best_guess,
            "best_guess_correct": best_guess_is_correct,
            "closest_alternative": closest_alternative,
            "closest_alternative_correct": closest_alternative_is_correct,
            "top2_hit": (best_guess_is_correct or closest_alternative_is_correct),
            "confidence_level": confidence_level,
            "trust_level": trust_level if run_trust_gating else None,
            "trust_high": trust_high if run_trust_gating else None,
            "pred_hcqr": predicted_hcqr,
            "hcqr_correct": hcqr_is_correct,
            "pred_no_rag": predicted_no_rag,
            "no_rag_correct": no_rag_is_correct,
            "pred_simple_rag": predicted_simple_rag,
            "simple_rag_correct": simple_rag_is_correct,
            "retrieve_for_self_conf": bool(retrieve_for_self_conf) if run_self_conf_gating else None,
            "pred_self_conf_gate": pred_self_conf_gate if run_self_conf_gating else "",
            "self_conf_gate_correct": self_conf_gate_correct if run_self_conf_gating else None,
            "retrieve_for_trust": bool(retrieve_for_trust) if run_trust_gating else None,
            "pred_trust_gate": pred_trust_gate if run_trust_gating else "",
            "trust_gate_correct": trust_gate_correct if run_trust_gating else None,
        })

    elapsed = time.time() - t_total
    hcqr_accuracy = hcqr_correct / total * 100 if total > 0 else 0
    no_rag_accuracy = no_rag_correct / total * 100 if total > 0 else 0
    simple_rag_accuracy = simple_rag_correct / total * 100 if total > 0 else 0
    self_conf_gating_accuracy = (
        self_conf_gating_correct / total * 100 if total > 0 else 0
    )
    trust_gating_accuracy = (
        trust_gating_correct / total * 100 if total > 0 else 0
    )
    closest_alternative_accuracy = (
        closest_alternative_correct / total * 100 if total > 0 else 0
    )
    top2_accuracy = (
        (best_guess_correct + closest_alternative_rescue) / total * 100 if total > 0 else 0
    )
    self_conf_retrieval_usage_ratio = (
        self_conf_retrieval_used_count / total if total > 0 else 0.0
    )
    trust_retrieval_usage_ratio = (
        trust_retrieval_used_count / total if total > 0 else 0.0
    )

    summary = {
        "config": {
            "llm_provider": "openai",
            "base_model": base_model,
            "hypothesis_checkpoint": hypothesis_checkpoint,
            "rewriter_checkpoint": rewriter_checkpoint,
            "generator_checkpoint": generator_checkpoint,
            "hypothesis_prompt": hypothesis_prompt,
            "rewriting_prompt": rewriting_prompt,
            "generator_prompt": generator_prompt,
            "question_dataset": question_dataset,
            "combo_tag": combo_tag,
            "total_evaluated": total,
            "total_docs": total_docs,
            "max_tokens": max_tokens,
            "max_concurrent": max_concurrent,
            "api_base": api_base,
            "seed": seed,
            "command": " ".join(sys.argv),
            "measure_best_guess_only": measure_best_guess_only,
            "use_gating": use_gating,
            "gating_mode": gating_mode if use_gating else None,
            "gating_policy": gating_policy if run_self_conf_gating else None,
            "trust_threshold": trust_threshold if run_trust_gating else None,
            "trust_evaluator_model": trust_eval_model if run_trust_gating else None,
            "trust_evaluator_prompt": trust_evaluator_prompt if run_trust_gating else None,
            "hypothesis_finalizer_prompt": (
                hypothesis_finalizer_prompt if run_trust_gating else None
            ),
        },
        "timing": {
            "total_seconds": elapsed,
            "avg_per_question": elapsed / total if total else 0,
            "questions_per_minute": total / elapsed * 60 if elapsed > 0 else 0,
        },
        "planner_results": {
            "best_guess_accuracy": best_guess_accuracy,
            "best_guess_correct": best_guess_correct,
            "best_guess_total": total,
            "closest_alternative_accuracy": closest_alternative_accuracy,
            "closest_alternative_correct": closest_alternative_correct,
            "closest_alternative_rescue": closest_alternative_rescue,
            "top2_accuracy": top2_accuracy,
            "confidence_counts": confidence_counts,
            "confidence_ratios": {
                "1": confidence_counts[1] / total if total else 0.0,
                "2": confidence_counts[2] / total if total else 0.0,
                "3": confidence_counts[3] / total if total else 0.0,
            },
            "trust_counts": trust_counts,
            "trust_ratios": {
                "1": trust_counts[1] / total if total else 0.0,
                "2": trust_counts[2] / total if total else 0.0,
                "3": trust_counts[3] / total if total else 0.0,
            },
        },
        "mode_results": {
            combo_tag: {
                "correct": hcqr_correct,
                "total": total,
                "accuracy": hcqr_accuracy,
            }
        },
        "timestamp": datetime.now().isoformat(),
    }
    if use_gating:
        summary["mode_results"].update({
            "no_rag": {
                "correct": no_rag_correct,
                "total": total,
                "accuracy": no_rag_accuracy,
            },
            "simple_rag": {
                "correct": simple_rag_correct,
                "total": total,
                "accuracy": simple_rag_accuracy,
            },
        })
        if run_self_conf_gating:
            summary["mode_results"]["self_confidence_gating"] = {
                "policy": gating_policy,
                "correct": self_conf_gating_correct,
                "total": total,
                "accuracy": self_conf_gating_accuracy,
                "retrieval_used_count": self_conf_retrieval_used_count,
                "retrieval_usage_ratio": self_conf_retrieval_usage_ratio,
            }
            # Backward-compatible alias for prior scripts.
            summary["mode_results"]["gating"] = summary["mode_results"]["self_confidence_gating"]
        if run_trust_gating:
            summary["mode_results"]["trust_gating"] = {
                "threshold": trust_threshold,
                "correct": trust_gating_correct,
                "total": total,
                "accuracy": trust_gating_accuracy,
                "retrieval_used_count": trust_retrieval_used_count,
                "retrieval_usage_ratio": trust_retrieval_usage_ratio,
            }

    print(f"\n{'='*70}")
    print(
        f"[HCQR:{combo_tag}] {hcqr_accuracy:.2f}% ({hcqr_correct}/{total}) "
        f"in {elapsed:.0f}s ({summary['timing']['questions_per_minute']:.1f} q/min)"
    )
    print(
        f"best_guess={best_guess_accuracy:.2f}% | "
        f"closest_alt={closest_alternative_accuracy:.2f}% | "
        f"top2={top2_accuracy:.2f}% | "
        f"conf(L1/L2/L3)=({confidence_counts[1]}/{confidence_counts[2]}/{confidence_counts[3]})"
    )
    if run_trust_gating:
        print(
            f"trust(L1/L2/L3)=({trust_counts[1]}/{trust_counts[2]}/{trust_counts[3]})"
        )
    if run_self_conf_gating:
        print(
            f"self_conf_gating({gating_policy})={self_conf_gating_accuracy:.2f}% | "
            f"no_rag={no_rag_accuracy:.2f}% | "
            f"simple_rag={simple_rag_accuracy:.2f}% | "
            f"retrieval_usage={self_conf_retrieval_usage_ratio*100:.1f}%"
        )
    if run_trust_gating:
        print(
            f"trust_gating(th={trust_threshold})={trust_gating_accuracy:.2f}% | "
            f"retrieval_usage={trust_retrieval_usage_ratio*100:.1f}%"
        )
    print(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", base_model).strip("_")
    if not model_suffix:
        model_suffix = "model"
    run_tag = combo_tag
    if use_gating:
        run_tag = f"{combo_tag}_gating_{gating_mode}"
        if run_self_conf_gating:
            run_tag = f"{run_tag}_{gating_policy}"
        if run_trust_gating:
            run_tag = f"{run_tag}_trust{trust_threshold}"
    out_file = os.path.join(
        output_dir,
        f"{question_dataset}_{run_tag}_{model_suffix}_{ts}.json",
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    out_jsonl = os.path.join(
        output_dir,
        f"{question_dataset}_{run_tag}_{model_suffix}_{ts}.jsonl",
    )
    out_csv = os.path.join(
        output_dir,
        f"{question_dataset}_{run_tag}_{model_suffix}_{ts}.csv",
    )
    _write_jsonl(out_jsonl, detail_rows)
    summary_csv_row = {
        "run_tag": run_tag,
        "model": base_model,
        "question_dataset": question_dataset,
        "hypothesis_prompt": hypothesis_prompt,
        "rewriting_prompt": rewriting_prompt,
        "generator_prompt": generator_prompt,
        "best_guess_accuracy": round(best_guess_accuracy, 4),
        "closest_alternative_accuracy": round(closest_alternative_accuracy, 4),
        "top2_accuracy": round(top2_accuracy, 4),
        "conf_1_ratio": round(confidence_counts[1] / total if total else 0.0, 6),
        "conf_2_ratio": round(confidence_counts[2] / total if total else 0.0, 6),
        "conf_3_ratio": round(confidence_counts[3] / total if total else 0.0, 6),
        "acc_hcqr": round(hcqr_accuracy, 4),
        "acc_no_rag": round(no_rag_accuracy, 4),
        "acc_simple_rag": round(simple_rag_accuracy, 4),
        "acc_self_conf_gating": round(self_conf_gating_accuracy if run_self_conf_gating else hcqr_accuracy, 4),
        "acc_trust_gating": round(trust_gating_accuracy if run_trust_gating else hcqr_accuracy, 4),
        "gating_mode": gating_mode if use_gating else "disabled",
        "gating_policy": gating_policy if run_self_conf_gating else "",
        "trust_threshold": trust_threshold if run_trust_gating else "",
        "trust_eval_model": trust_eval_model if run_trust_gating else "",
        "trust_prompt": trust_evaluator_prompt if run_trust_gating else "",
        "finalizer_prompt": hypothesis_finalizer_prompt if run_trust_gating else "",
        "retrieval_usage_self_conf": round(self_conf_retrieval_usage_ratio if run_self_conf_gating else 1.0, 6),
        "retrieval_usage_trust": round(trust_retrieval_usage_ratio if run_trust_gating else 1.0, 6),
        "trust_1_ratio": round(trust_counts[1] / total if total else 0.0, 6),
        "trust_2_ratio": round(trust_counts[2] / total if total else 0.0, 6),
        "trust_3_ratio": round(trust_counts[3] / total if total else 0.0, 6),
        "seed": seed,
        "command": " ".join(sys.argv),
    }
    _write_summary_csv(out_csv, summary_csv_row)

    print(f"✓ Saved: {out_file}")
    print(f"✓ Saved details jsonl: {out_jsonl}")
    print(f"✓ Saved summary csv: {out_csv}")

    return {
        "combo": run_tag,
        "accuracy": (
            trust_gating_accuracy
            if run_trust_gating
            else self_conf_gating_accuracy
            if run_self_conf_gating
            else hcqr_accuracy
        ),
        "correct": (
            trust_gating_correct
            if run_trust_gating
            else self_conf_gating_correct
            if run_self_conf_gating
            else hcqr_correct
        ),
        "total": total,
        "file": out_file,
    }


def run_batch_phased_evaluation(
    base_model: str,
    rewriter_checkpoint: str = None,
    hypothesis_checkpoint: str = None,
    generator_checkpoint: str = None,
    max_questions: int = 1273,
    total_docs: int = 15,
    question_dataset: str = "medqa",
    gpu_mem: float = 0.9,
    max_model_len: int = 8192,
    max_tokens: int = 2048,
    output_dir: Optional[str] = None,
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    vllm_tensor_parallel_size: int = 1,
    hypothesis_prompt: str = "v1",
    rewriting_prompt: str = "v1",
    generator_prompt: str = "v1",
) -> Dict[str, Any]:
    """
    Batch-phased evaluation with selectable prompts.
    Each vLLM phase runs in a separate subprocess → full GPU memory release.

    Args:
        hypothesis_prompt: "v1", "v2", or "v3" (from core/prompts.py)
        rewriting_prompt:  "v1", "v2", or "v3" (from core/prompts.py)
        generator_prompt:  "v1", "v2", or "v3" (from core/prompts.py)
    """
    output_dir = resolve_results_output_dir(
        output_dir=output_dir,
        llm_provider="vllm",
        model_name=base_model,
    )
    set_global_seed(EVAL_FIXED_SEED)

    from training.reward import (
        parse_queries_from_completion, parse_hypothesis_plan,
    )

    # Resolve per-module models
    hypothesis_model = hypothesis_checkpoint or base_model
    rewriter_model = rewriter_checkpoint or base_model
    generator_model = generator_checkpoint or base_model

    h_prompt = HYPOTHESIS_PROMPTS[hypothesis_prompt]
    r_prompt = REWRITING_PROMPTS[rewriting_prompt]
    g_prompt = GENERATOR_PROMPTS[generator_prompt]
    combo_tag = f"h{hypothesis_prompt}_r{rewriting_prompt}_g{generator_prompt}"

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = MedQADataset(question_dataset=question_dataset)
    capped_total = max(0, min(max_questions, len(dataset)))
    question_ids = list(range(capped_total))
    questions = [dataset[i] for i in question_ids]
    n = len(questions)
    print(f"\n{'='*70}")
    print(f"Batch-Phased Evaluation: {n} questions")
    print(f"Question set: {question_dataset}")
    print(f"Combo:        {combo_tag}")
    print(f"  hypothesis: {h_prompt['description']}")
    print(f"  rewriting:  {r_prompt['description']}")
    print(f"  generator:  {g_prompt['description']}")
    print(f"Base model:   {base_model}")
    print(f"Hypothesis:   {hypothesis_model}"
          f"{' (checkpoint)' if hypothesis_checkpoint else ''}")
    print(f"Rewriter:     {rewriter_model}"
          f"{' (checkpoint)' if rewriter_checkpoint else ''}")
    print(f"Generator:    {generator_model}"
          f"{' (checkpoint)' if generator_checkpoint else ''}")
    print(f"vllm_tp={vllm_tensor_parallel_size}  "
          f"gpu_mem={gpu_mem}  max_model_len={max_model_len}  "
          f"max_tokens={max_tokens}")
    print(f"seed={EVAL_FIXED_SEED}  temperature={EVAL_FIXED_TEMPERATURE}")
    print(f"do_sample={EVAL_FIXED_DO_SAMPLE} (where supported)")
    print(f"{'='*70}")

    t_total = time.time()

    # ── Phase 1: Hypothesis ALL (subprocess) ──
    print(f"\n[Phase 1/4] Generating hypotheses for {n} questions "
          f"({os.path.basename(hypothesis_model)}) ...")
    t1 = time.time()

    plan_messages = []
    for qd in questions:
        options_text = "\n".join(
            [f"{k}. {v}" for k, v in sorted(qd["options"].items())])
        user = h_prompt["user"].format(
            question=qd["question"], options=options_text)
        plan_messages.append([
            {"role": "system", "content": h_prompt["system"]},
            {"role": "user", "content": user},
        ])

    plan_texts = _run_phase_in_subprocess(
        hypothesis_model, plan_messages, gpu_mem, max_model_len, max_tokens,
        vllm_tensor_parallel_size)

    plans = []
    best_guess_correct = 0
    closest_alternative_correct = 0
    closest_alternative_rescue = 0
    for idx, text in enumerate(plan_texts):
        p = parse_hypothesis_plan(text)
        p.setdefault("discriminating_features", [])
        p.setdefault("best_guess", "")
        p.setdefault("best_guess_text", "")
        p.setdefault("reasoning", "")
        p.setdefault("confirming_evidence", [])
        p.setdefault("alternative_if_wrong", "")
        p.setdefault("closest_alternative", "")
        p.setdefault("closest_alternative_text", "")
        p["confidence_level"] = coerce_confidence_level(
            p.get("confidence_level", 2),
            default=2,
        )
        if not p["discriminating_features"]:
            words = re.findall(
                r'\b[A-Za-z]{4,}\b', questions[idx]["question"])
            p["discriminating_features"] = words[:3] or ["symptom"]
        p = _enrich_plan_with_alternatives(
            p,
            text,
            questions[idx].get("options", {}),
        )
        gold = coerce_choice(questions[idx].get("answer_idx", questions[idx].get("answer", "")))
        best_guess = coerce_choice(p.get("best_guess", ""))
        closest_alternative = coerce_choice(
            p.get("closest_alternative", p.get("alternative_if_wrong", ""))
        )
        if best_guess == gold and bool(gold):
            best_guess_correct += 1
        if closest_alternative == gold and bool(gold):
            closest_alternative_correct += 1
            if best_guess != gold:
                closest_alternative_rescue += 1
        plans.append(p)

    print(f"  ✓ Phase 1 done: {len(plans)} hypotheses "
          f"in {time.time()-t1:.1f}s")
    best_guess_accuracy = (best_guess_correct / n * 100) if n > 0 else 0.0
    closest_alternative_accuracy = (
        closest_alternative_correct / n * 100 if n > 0 else 0.0
    )
    top2_accuracy = (
        (best_guess_correct + closest_alternative_rescue) / n * 100 if n > 0 else 0.0
    )
    print(
        f"  ✓ best_guess={best_guess_accuracy:.2f}% | "
        f"closest_alt={closest_alternative_accuracy:.2f}% | "
        f"top2={top2_accuracy:.2f}%"
    )

    # ── Phase 2: Rewrite ALL (rewriter model, subprocess) ──
    print(f"\n[Phase 2/4] Generating queries for {n} questions "
          f"({os.path.basename(rewriter_model)}) ...")
    t2 = time.time()

    rw_messages = []
    for i, qd in enumerate(questions):
        p = plans[i]
        options_text = "\n".join(
            [f"{k}. {v}" for k, v in sorted(qd["options"].items())])
        # Expand best_guess letter to include option text (e.g. "A" -> "A. Psoriasis")
        bg_letter = (p.get("best_guess", "") or "").strip().upper().rstrip(".")
        # Prefer model-generated best_guess_text (hv7+), fall back to code expansion
        bg_text = p.get("best_guess_text") or (
            f"{bg_letter}. {qd['options'].get(bg_letter, bg_letter)}" if bg_letter in qd.get('options', {}) else bg_letter
        )
        alt_letter = (
            p.get("closest_alternative", "")
            or p.get("alternative_if_wrong", "")
            or ""
        ).strip().upper().rstrip(".")
        closest_alternative_text = p.get("closest_alternative_text") or ""
        alt_text = closest_alternative_text or (
            f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
            if alt_letter in qd.get('options', {}) else alt_letter
        )
        user = r_prompt["user"].format(
            question=qd["question"],
            options=options_text,
            best_guess=p.get("best_guess", ""),
            best_guess_text=bg_text,
            reasoning=p.get("reasoning", ""),
            confirming_evidence=_format_list(
                p.get("confirming_evidence", [])),
            discriminating_features=_format_list(
                p.get("discriminating_features", [])),
            alternative_if_wrong=p.get("alternative_if_wrong", ""),
            alternative_text=alt_text,
            closest_alternative=p.get("closest_alternative", ""),
            closest_alternative_text=alt_text,
        )
        rw_messages.append([
            {"role": "system", "content": r_prompt["system"]},
            {"role": "user", "content": user},
        ])

    rw_texts = _run_phase_in_subprocess(
        rewriter_model, rw_messages, gpu_mem, max_model_len, max_tokens,
        vllm_tensor_parallel_size)

    all_queries = []
    for i, text in enumerate(rw_texts):
        queries = parse_queries_from_completion(text)
        if len(queries) < 2:
            p = plans[i]
            bg = p.get("best_guess", "")
            cf = _format_list(p.get("confirming_evidence", []))
            df = _format_list(p.get("discriminating_features", []))
            queries = [f"{bg} {df}", cf,
                       f"{p.get('reasoning','')} diagnosis"]
        all_queries.append(queries[:5])

    print(f"  ✓ Phase 2 done: {len(all_queries)} query sets "
          f"in {time.time()-t2:.1f}s")

    # ── Phase 3: Retrieve ALL (CPU, in-process) ──
    print(f"\n[Phase 3/4] Retrieving documents for {n} questions ...")
    t3 = time.time()

    try:
        from retrieval.retriever import create_retriever as _cr
        retriever = _cr(
            retriever_type="mirage",
            retriever_name=retriever_name,
            corpus_name=corpus_name,
        )
        if hasattr(retriever, "_lazy_init"):
            retriever._lazy_init()
    except Exception as e:
        print(f"  WARNING: Retriever init failed: {e}")
        retriever = None

    all_docs = []
    for i, queries in enumerate(all_queries):
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        if retriever:
            k_per = max(1, total_docs // max(len(queries), 1))
            for q in queries:
                try:
                    docs, scores = retriever.retrieve(q, k=k_per)
                    for doc, score in zip(docs, scores):
                        doc_id = doc.get(
                            "id",
                            doc.get("title",
                                    str(hash(doc.get("content", "")[:100]))))
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = 0.0
                            doc_data[doc_id] = doc.copy()
                            doc_data[doc_id]["query_trace"] = []
                        try:
                            doc_scores[doc_id] += float(score)
                        except Exception:
                            doc_scores[doc_id] += 0.0
                        doc_data[doc_id]["query_trace"].append(q)
                except Exception:
                    pass

        docs_sorted = []
        for doc_id in sorted(
                doc_scores.keys(),
                key=lambda x: doc_scores[x], reverse=True):
            doc = doc_data[doc_id]
            doc["fused_score"] = doc_scores[doc_id]
            docs_sorted.append(doc)

        all_docs.append(docs_sorted)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{n} retrieved")

    print(f"  ✓ Phase 3 done: retrieval in {time.time()-t3:.1f}s")

    # Free retriever GPU memory before Phase 4 vLLM subprocess
    if retriever is not None:
        del retriever
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    print("  ✓ Retriever GPU memory freed")

    # ── Phase 4: Answer ALL (generator model, subprocess) ──
    print(f"\n[Phase 4/4] Generating answers for {n} questions "
          f"({os.path.basename(generator_model)}) ...")
    t4 = time.time()

    ans_messages = []
    for i, qd in enumerate(questions):
        options_text = "\n".join(
            [f"{k}. {v}" for k, v in sorted(qd["options"].items())])
        docs = all_docs[i]
        ctx = "\n\n".join(
            [f"Document [{j+1}] (Title: {d.get('title','Untitled')})\n"
             f"{d.get('content','')}"
             for j, d in enumerate(docs[:25])]
        ) or "No documents."

        fmt_vars = {
            "context": ctx,
            "question": qd["question"],
            "options": options_text,
            "hypothesis_summary": "",
            "queries_summary": "",
            "best_guess": "",
            "best_guess_text": "",
            "closest_alternative": "",
            "closest_alternative_text": "",
        }
        # generator v2/v3 needs hypothesis context
        if generator_prompt in {"v2", "v3"}:
            p = plans[i]
            bg_letter = (p.get("best_guess", "") or "").strip().upper().rstrip(".")
            bg_text = p.get("best_guess_text") or (
                f"{bg_letter}. {qd['options'].get(bg_letter, bg_letter)}"
                if bg_letter in qd.get("options", {})
                else bg_letter
            )
            alt_letter = (
                p.get("closest_alternative", "")
                or p.get("alternative_if_wrong", "")
                or ""
            ).strip().upper().rstrip(".")
            closest_alternative_text = p.get("closest_alternative_text") or ""
            alt_text = closest_alternative_text or (
                f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
                if alt_letter in qd.get("options", {})
                else alt_letter
            )
            fmt_vars["hypothesis_summary"] = (
                f"Best guess: {p.get('best_guess','')} — "
                f"{p.get('reasoning','')}"
            )
            fmt_vars["queries_summary"] = "\n".join(
                f"  {j+1}. {q}" for j, q in enumerate(all_queries[i]))
            fmt_vars["best_guess"] = p.get("best_guess", "")
            fmt_vars["best_guess_text"] = bg_text
            fmt_vars["closest_alternative"] = p.get("closest_alternative", "")
            fmt_vars["closest_alternative_text"] = alt_text

        user = g_prompt["user"].format(**fmt_vars)
        ans_messages.append([
            {"role": "system", "content": g_prompt["system"]},
            {"role": "user", "content": user},
        ])

    ans_texts = _run_phase_in_subprocess(
        generator_model, ans_messages, gpu_mem, max_model_len, max_tokens,
        vllm_tensor_parallel_size)

    print(f"  ✓ Phase 4 done: answers in {time.time()-t4:.1f}s")

    # ── Compile Results ──
    print(f"\n[Result] Compiling ...")
    results = []
    correct = 0
    total = 0
    for i, qd in enumerate(questions):
        correct_answer = qd.get("answer_idx", qd.get("answer", ""))
        raw_resp = ans_texts[i]
        predicted = parse_answer(raw_resp)
        is_correct = predicted.upper() == correct_answer.upper()
        if is_correct:
            correct += 1
        total += 1
        results.append({
            "question_id": question_ids[i],
            "question": qd["question"],
            "options": qd["options"],
            "correct_answer": correct_answer,
            "modes": {
                combo_tag: {
                    "num_queries": len(all_queries[i]),
                    "num_docs": len(all_docs[i]),
                    "queries": all_queries[i],
                    "plan": plans[i],
                    "retrieved_docs": _serialize_retrieved_docs_for_output(all_docs[i]),
                    "raw_response": raw_resp,
                    "predicted_answer": predicted,
                    "is_correct": is_correct,
                }
            },
        })

    elapsed = time.time() - t_total
    accuracy = correct / total * 100 if total > 0 else 0

    summary = {
        "config": {
            "base_model": base_model,
            "hypothesis_checkpoint": hypothesis_checkpoint,
            "rewriter_checkpoint": rewriter_checkpoint,
            "generator_checkpoint": generator_checkpoint,
            "hypothesis_prompt": hypothesis_prompt,
            "rewriting_prompt": rewriting_prompt,
            "generator_prompt": generator_prompt,
            "question_dataset": question_dataset,
            "combo_tag": combo_tag,
            "total_evaluated": total,
            "total_docs": total_docs,
            "gpu_mem": gpu_mem,
            "vllm_tensor_parallel_size": vllm_tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
        },
        "timing": {
            "total_seconds": elapsed,
            "avg_per_question": elapsed / total if total else 0,
            "questions_per_minute":
                total / elapsed * 60 if elapsed > 0 else 0,
        },
        "planner_results": {
            "best_guess_accuracy": best_guess_accuracy,
            "best_guess_correct": best_guess_correct,
            "best_guess_total": total,
            "closest_alternative_accuracy": closest_alternative_accuracy,
            "closest_alternative_correct": closest_alternative_correct,
            "closest_alternative_rescue": closest_alternative_rescue,
            "top2_accuracy": top2_accuracy,
        },
        "mode_results": {
            combo_tag: {
                "correct": correct, "total": total, "accuracy": accuracy,
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*70}")
    print(f"[{combo_tag}] {accuracy:.2f}% ({correct}/{total}) "
          f"in {elapsed:.0f}s "
          f"({summary['timing']['questions_per_minute']:.1f} q/min)")
    print(
        f"best_guess={best_guess_accuracy:.2f}% | "
        f"closest_alt={closest_alternative_accuracy:.2f}% | "
        f"top2={top2_accuracy:.2f}%"
    )
    print(f"{'='*70}")

    # Save — name encodes prompt combination
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(
        output_dir,
        f"{question_dataset}_{combo_tag}_{ts}.json",
    )
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {out_file}")

    return {"combo": combo_tag, "accuracy": accuracy,
            "correct": correct, "total": total, "file": out_file}


def run_hypothesis_combo(
    args: argparse.Namespace,
    corpus_name: str,
    model_name: str,
    hypothesis_prompt: str,
    rewriting_prompt: str,
    generator_prompt: str,
) -> Dict[str, Any]:
    """Execute one hypothesis-mode run (OpenAI or local vLLM backend)."""
    if args.llm_provider == "openai":
        return asyncio.run(run_batch_phased_evaluation_openai(
            base_model=model_name,
            rewriter_checkpoint=args.rewriter_checkpoint,
            hypothesis_checkpoint=args.hypothesis_checkpoint,
            max_questions=args.max_questions,
            total_docs=args.total_docs,
            question_dataset=args.question_dataset,
            max_tokens=args.vllm_max_tokens,
            max_concurrent=args.max_concurrent,
            output_dir=args.output_dir,
            retriever_name=args.retriever,
            corpus_name=corpus_name,
            hypothesis_prompt=hypothesis_prompt,
            rewriting_prompt=rewriting_prompt,
            generator_prompt=generator_prompt,
            api_base=args.api_base,
            measure_best_guess_only=args.measure_best_guess_only,
            use_gating=args.use_gating,
            gating_mode=args.gating_mode,
            gating_policy=args.gating_policy,
            trust_threshold=args.trust_threshold,
            trust_evaluator_model=args.trust_evaluator_model,
            trust_evaluator_prompt=args.trust_evaluator_prompt,
            hypothesis_finalizer_prompt=args.hypothesis_finalizer_prompt,
            seed=EVAL_FIXED_SEED,
        ))

    if args.measure_best_guess_only or args.use_gating:
        raise ValueError(
            "best_guess_only / gating options are currently supported only with --llm-provider openai."
        )

    return run_batch_phased_evaluation(
        base_model=model_name,
        rewriter_checkpoint=args.rewriter_checkpoint,
        hypothesis_checkpoint=args.hypothesis_checkpoint,
        max_questions=args.max_questions,
        total_docs=args.total_docs,
        question_dataset=args.question_dataset,
        gpu_mem=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_tokens=args.vllm_max_tokens,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        output_dir=args.output_dir,
        retriever_name=args.retriever,
        corpus_name=corpus_name,
        hypothesis_prompt=hypothesis_prompt,
        rewriting_prompt=rewriting_prompt,
        generator_prompt=generator_prompt,
    )


def resolve_hypothesis_combo_triplets(
    args: argparse.Namespace,
) -> List[Tuple[str, str, str]]:
    """Resolve hypothesis/rewriter/generator prompt combinations for ablation."""
    if args.run_all:
        return [
            (hypothesis_version, rewrite_version, "v1")
            for hypothesis_version in ["v1", "v2", "v3"]
            for rewrite_version in ["v1", "v2", "v3"]
        ]

    combo_tokens = args.ablation_combos or [
        f"{args.hypothesis_prompt}-{args.rewriting_prompt}-{args.generator_prompt}"
    ]
    return parse_ablation_combo_tokens(
        combo_tokens,
        default_generator_prompt=args.generator_prompt,
    )


def save_ablation_summary_csv(
    summary_rows: List[Dict[str, Any]],
    output_dir: Optional[str],
    llm_provider: str,
) -> str:
    """Save aggregate ablation metrics as CSV and return file path."""
    import csv

    summary_dir = resolve_ablation_summary_dir(
        output_dir=output_dir,
        llm_provider=llm_provider,
    )
    os.makedirs(summary_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(summary_dir, f"ablation_summary_{timestamp}.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "combo", "accuracy", "correct", "total", "file"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return csv_file


def run_hypothesis_mode(args: argparse.Namespace, corpus_name: str) -> None:
    """Entry point for hypothesis mode: single combo or multi-run ablation."""
    run_ablation = bool(
        args.run_all or args.ablation_models is not None
        or args.ablation_combos is not None
    )

    if not run_ablation:
        run_hypothesis_combo(
            args=args,
            corpus_name=corpus_name,
            model_name=args.model,
            hypothesis_prompt=args.hypothesis_prompt,
            rewriting_prompt=args.rewriting_prompt,
            generator_prompt=args.generator_prompt,
        )
        return

    combo_triplets = resolve_hypothesis_combo_triplets(args)
    model_names = args.ablation_models or [args.model]
    summary_rows = []

    total_runs = len(model_names) * len(combo_triplets)
    run_idx = 0

    print(f"\n{'#'*70}")
    print(f"Running ablation: {len(model_names)} model(s) x "
          f"{len(combo_triplets)} combo(s) = {total_runs} run(s)")
    print(f"{'#'*70}")

    for model_name in model_names:
        for hv, rv, gv in combo_triplets:
            run_idx += 1
            print(f"\n\n{'#'*70}")
            print(
                f"  Run {run_idx}/{total_runs}: model={model_name}, "
                f"hypothesis={hv}, rewriting={rv}, generator={gv}"
            )
            print(f"{'#'*70}")
            result = run_hypothesis_combo(
                args=args,
                corpus_name=corpus_name,
                model_name=model_name,
                hypothesis_prompt=hv,
                rewriting_prompt=rv,
                generator_prompt=gv,
            )
            summary_rows.append({
                "model": model_name,
                **result,
            })

    print(f"\n\n{'='*90}")
    print("ABLATION SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<28} {'Combo':<15} {'Accuracy':>10} {'Correct':>10} {'Total':>8}")
    print(f"{'-'*28} {'-'*15} {'-'*10} {'-'*10} {'-'*8}")
    for row in summary_rows:
        print(f"{_slugify(row['model']):<28} {row['combo']:<15} "
              f"{row['accuracy']:>9.2f}% {row['correct']:>10} {row['total']:>8}")
    print(f"{'='*90}")

    csv_file = save_ablation_summary_csv(
        summary_rows=summary_rows,
        output_dir=args.output_dir,
        llm_provider=args.llm_provider,
    )
    print(f"✓ Summary CSV saved: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='MedQA RAG Evaluation v2 (Prompt-Selectable Ablation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Hypothesis mode with specific prompts
  python scripts/evaluate/evaluate.py --mode hypothesis \\
      --hypothesis-prompt v1 --rewriting-prompt v2 --generator-prompt v1 \\
      --model Qwen/Qwen3-4B-Instruct-2507 --max-questions 1273

  # Run all 9 hypothesis×rewriting combos (generator=v1)
  python scripts/evaluate/evaluate.py --mode hypothesis --run-all \\
      --model Qwen/Qwen3-4B-Instruct-2507 --max-questions 1273

  # With checkpoints
  python scripts/evaluate/evaluate.py --mode hypothesis \\
      --hypothesis-prompt v1 --rewriting-prompt v1 \\
      --model Qwen/Qwen3-4B-Instruct-2507 \\
      --hypothesis-checkpoint /path/to/checkpoint-1100

  # Best-guess only (planner) with v7
  python scripts/evaluate/evaluate.py --mode hypothesis \\
      --llm-provider openai --model gpt-4o-mini \\
      --hypothesis-prompt v7 --measure-best-guess-only

  # Direct gating (v7plus self-confidence)
  python scripts/evaluate/evaluate.py --mode hypothesis \\
      --llm-provider openai --model gpt-4o-mini \\
      --gating direct_gating --rewriting-prompt v10 --generator-prompt v1

  # Agentic gating (v7 + evaluator trust)
  python scripts/evaluate/evaluate.py --mode hypothesis \\
      --llm-provider openai --model gpt-4o-mini \\
      --gating agentic_gating --rewriting-prompt v10 --generator-prompt v1

  # Baseline modes (cot, directrag, directrewriting)
  python scripts/evaluate/evaluate.py --mode cot --max-questions 1273
  python scripts/evaluate/evaluate.py --mode directrag --max-questions 1273
  python scripts/evaluate/evaluate.py --mode directrewriting --max-questions 1273

  # Prompt/model ablation (replaces old run_gpt_ablation.py)
  python scripts/evaluate/evaluate.py --mode hypothesis --llm-provider openai \
      --ablation-models gpt-4o-mini gpt-4o \
      --ablation-combos v5-v5-v2 v7-v10-v2 \
      --max-questions 1273
        """
    )

    # ── Mode selection ──
    parser.add_argument(
        '--mode', type=str, default='hypothesis',
        choices=['cot', 'directrag', 'directrewriting', 'hypothesis', 'direct', 'baseline'],
        help=('Evaluation mode (default: hypothesis). '
              'Legacy aliases: direct->directrag, baseline->directrewriting'))
    parser.add_argument(
        '--hypothesis-prompt', type=str, default='v1',
        choices=sorted(HYPOTHESIS_PROMPTS.keys()),
        help='Hypothesis prompt version (default: v1)')
    parser.add_argument(
        '--rewriting-prompt', type=str, default='v1',
        choices=sorted(REWRITING_PROMPTS.keys()),
        help='Rewriting prompt version (default: v1)')
    parser.add_argument(
        '--generator-prompt', type=str, default='v1',
        choices=sorted(GENERATOR_PROMPTS.keys()),
        help='Generator prompt version (default: v1)')
    parser.add_argument(
        '--run-all', action='store_true',
        help='Run all 9 hypothesis×rewriting combos with generator=v1')
    parser.add_argument(
        '--ablation-models', nargs='+', default=None,
        help='Run multiple models in one command (hypothesis mode only).')
    parser.add_argument(
        '--ablation-combos', nargs='+', default=None,
        help=("Prompt combos for hypothesis mode: 'vH-vR' or 'vH-vR-vG' "
              "(e.g., v5-v5-v2 v7-v10-v2)."))
    parser.add_argument(
        '--measure-best-guess-only', action='store_true',
        help='Hypothesis mode only: run planner phase and report best_guess/confidence metrics only.'
    )
    parser.add_argument(
        '--gating', type=str, default=None,
        choices=['direct_gating', 'agentic_gating'],
        help=('Hypothesis mode only: gating preset. '
              'direct_gating = v7plus self-confidence gating; '
              'agentic_gating = v7 + trust evaluator gating. '
              'If omitted, gating is disabled.')
    )
    # Backward-compatible flags (hidden)
    parser.add_argument(
        '--use-gating', action='store_true',
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--gating-mode', type=str, default='both',
        choices=['self_confidence', 'trust_eval', 'both'],
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--gating-policy', type=str, default='conf3_skip',
        choices=['conf3_skip', 'conf23_skip', 'conf3_only_retrieve'],
        help=('Self-confidence gating policy when --use-gating and '
              '--gating-mode includes self_confidence. '
              'conf3_skip: retrieve on conf 1/2; '
              'conf23_skip: retrieve on conf 1 only; '
              'conf3_only_retrieve: retrieve on conf 3 only.')
    )
    parser.add_argument(
        '--trust-threshold', type=int, default=3, choices=[1, 2, 3],
        help=('Trust gating threshold when --gating-mode includes trust_eval. '
              'high-trust is trust_level >= threshold (default: 3).')
    )
    parser.add_argument(
        '--trust-evaluator-model', type=str, default=None,
        help='Model for trust evaluator phase (default: --model).'
    )
    parser.add_argument(
        '--trust-evaluator-prompt', type=str, default='v1',
        choices=sorted(TRUST_EVALUATOR_PROMPTS.keys()),
        help='Trust evaluator prompt version (default: v1).'
    )
    parser.add_argument(
        '--hypothesis-finalizer-prompt', type=str, default='v1',
        choices=sorted(HYPOTHESIS_FINALIZER_PROMPTS.keys()),
        help='Hypothesis finalizer prompt version for high-trust branch (default: v1).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Ignored at runtime. Evaluation seed is fixed to 42.'
    )

    # ── Model & checkpoints ──
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--hypothesis-checkpoint', type=str, default=None,
                       help='Path to hypothesis checkpoint (Phase 1 model)')
    parser.add_argument('--rewriter-checkpoint', type=str, default=None,
                       help='Path to rewriter checkpoint (Phase 2 model)')
    parser.add_argument('--rewriter-base-model', type=str, default=None,
                       help='Base model for rewriter (defaults to --model)')

    # ── Evaluation settings ──
    parser.add_argument('--max-questions', '-n', type=int, default=1273)
    parser.add_argument('--max-concurrent', '-c', type=int, default=100)
    parser.add_argument('--llm-provider', type=str, default='openai',
                       choices=['openai', 'vllm'])
    parser.add_argument('--retriever', type=str, default='MedCPT')
    parser.add_argument(
        '--retrieval-dataset',
        type=parse_retrieval_dataset,
        default=RETRIEVAL_DATASET_TO_CORPUS["textbooks"],
        metavar='{textbooks,pubmed}',
        help='Retrieval dataset (default: textbooks)')
    parser.add_argument(
        '--question-dataset',
        type=parse_question_dataset,
        default='medqa',
        metavar='{' + ",".join(QUESTION_DATASET_CHOICES) + '}',
        help='Question dataset in benchmark.json (default: medqa)')
    parser.add_argument('--corpus', type=str, default=None,
                       help=argparse.SUPPRESS)
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='Optional output directory. If omitted, uses outputs/results/<provider>/<model>/'
    )
    parser.add_argument('--api-base', type=str, default=None)
    parser.add_argument('--total-docs', type=int, default=15)

    # ── vLLM settings ──
    parser.add_argument('--vllm-tensor-parallel-size', type=int, default=1)
    parser.add_argument('--vllm-gpu-memory-utilization', type=float,
                       default=0.9)
    parser.add_argument('--vllm-max-tokens', type=int, default=2048)
    parser.add_argument('--vllm-max-concurrent', type=int, default=1)
    parser.add_argument('--vllm-max-model-len', type=int, default=8192)
    parser.add_argument('--question-timeout-seconds', type=int, default=600)

    args = parser.parse_args()
    args.mode = normalize_mode_name(args.mode)

    # Normalize gating preset into internal legacy flags.
    run_ablation_requested = bool(
        args.run_all or args.ablation_models is not None or args.ablation_combos is not None
    )
    if args.gating is not None and run_ablation_requested:
        raise ValueError("--gating cannot be combined with --run-all/--ablation-* options.")

    if args.gating == "direct_gating":
        args.use_gating = True
        args.gating_mode = "self_confidence"
        if args.hypothesis_prompt != "v7plus":
            print("INFO: --gating direct_gating uses v7plus planner. Overriding --hypothesis-prompt to v7plus.")
            args.hypothesis_prompt = "v7plus"
    elif args.gating == "agentic_gating":
        args.use_gating = True
        args.gating_mode = "trust_eval"
        if args.hypothesis_prompt != "v7":
            print("INFO: --gating agentic_gating uses v7 planner. Overriding --hypothesis-prompt to v7.")
            args.hypothesis_prompt = "v7"

    if args.seed != EVAL_FIXED_SEED:
        print(f"INFO: --seed {args.seed} ignored; using fixed seed={EVAL_FIXED_SEED}")
    args.seed = EVAL_FIXED_SEED
    set_global_seed(args.seed)
    corpus_name = args.corpus if args.corpus else args.retrieval_dataset

    if args.mode != "hypothesis" and (args.use_gating or args.measure_best_guess_only or args.gating is not None):
        raise ValueError(
            "--gating/--use-gating and --measure-best-guess-only are supported only with --mode hypothesis."
        )

    if args.mode == 'hypothesis':
        run_hypothesis_mode(args, corpus_name)
        return

    # ── Baseline modes (cot, directrag, directrewriting) via async evaluator ──
    asyncio.run(run_evaluation_async(
        max_questions=args.max_questions,
        llm_provider=args.llm_provider,
        model_name=args.model,
        retriever_name=args.retriever,
        corpus_name=corpus_name,
        question_dataset=args.question_dataset,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        modes=[args.mode],
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_tokens=args.vllm_max_tokens,
        vllm_max_concurrent=args.vllm_max_concurrent,
        vllm_max_model_len=args.vllm_max_model_len,
        question_timeout_seconds=args.question_timeout_seconds,
        total_docs=args.total_docs,
        # Baseline path uses the base model for query rewriting.
        rewriter_adapter_path=None,
        rewriter_base_model=None,
        api_base=args.api_base,
    ))


if __name__ == "__main__":
    import multiprocessing as _main_mp
    _main_mp.set_start_method("spawn", force=True)
    main()
