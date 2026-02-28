#!/usr/bin/env python3
"""
MedQA Evaluation v2 – Prompt-Selectable Ablation System

Modes:
  cot       : Chain-of-Thought (no RAG)
  direct    : 1 query × total_docs
  baseline  : 5 sub-queries × (total_docs/5)
  hypothesis: Hypothesis→Rewrite→Retrieve→Answer (4-phase batch)

For hypothesis mode, select prompt versions:
  --hypothesis-prompt  v1 | v2 | v3
  --rewriting-prompt   v1 | v2 | v3
  --generator-prompt   v1 | v2
  --run-all  runs all 9 hypothesis×rewriting combos (generator=v1)
"""

import os
import sys
import json
import time
import re
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import aiohttp

# Add current directory to path
# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from retriever import create_retriever
from prompts import (
    HYPOTHESIS_PROMPTS, REWRITING_PROMPTS, GENERATOR_PROMPTS,
    MULTI_QUERY_PROMPT_5, PLANNING_PROMPT,
    load_mirage_prompts,
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

# Load MIRAGE prompts (overrides defaults in prompts.py if available)
if load_mirage_prompts():
    print("✓ Loaded MIRAGE prompts from template.py")
else:
    print("⚠ Using built-in prompts")

# Re-import after potential override
from prompts import COT_SYSTEM_PROMPT, MIRAGE_SYSTEM_PROMPT


# ============================================================================
# MedQA Dataset Loader
# ============================================================================
class MedQADataset:
    """MedQA Dataset loader"""
    
    def __init__(self, benchmark_path: Optional[str] = None):
        if benchmark_path is None:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "MIRAGE", "benchmark.json"),
                os.path.join(os.path.dirname(__file__), "..", "MIRAGE", "benchmark.json"),
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
        
        self.dataset = benchmark["medqa"]
        self.index = sorted(self.dataset.keys())
        print(f"✓ Loaded {len(self)} MedQA questions")
    
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


def parse_subqueries(content: str, num_queries: int = 5) -> List[str]:
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
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }

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
        max_retries: int = 2,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 2048,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
    ):
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        # vLLM offline LLM is NOT thread-safe; serialize all calls
        self.semaphore = asyncio.Semaphore(1)

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
        )
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=["###", "User:", "\n\n\n"],
        )
        self._tokenizer = self._llm.get_tokenizer()

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
        if temperature != 0.0:
            sampling_params = self._SamplingParams(
                temperature=temperature,
                max_tokens=self.max_tokens,
                stop=["###", "User:", "\n\n\n"],
            )
        else:
            sampling_params = self._sampling_params
        outputs = self._llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0,
        session: aiohttp.ClientSession = None,
    ) -> str:
        """Generate text using local vLLM (session kept for API compatibility)"""
        del session
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    return await asyncio.to_thread(self._generate_sync, messages, temperature)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        return f"Error: {str(e)}"
                    await asyncio.sleep(1)
        return "Error: Max retries exceeded"


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
    
    def retrieve_documents(self, queries: List[str], k_per_query: int) -> List[Dict[str, Any]]:
        """Retrieve and fuse documents from multiple queries"""
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
        for doc_id in sorted_ids[:25]:
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
    ) -> List[str]:
        """Generate subqueries asynchronously"""
        
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
        
        prompt = MULTI_QUERY_PROMPT_5.format(query=enhanced_query)
        
        messages = [
            {"role": "system", "content": MIRAGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat_completion(messages, session=session)
        queries = parse_subqueries(response, self.num_subqueries)
        
        if len(queries) < 3:
            words = re.findall(r'\b[A-Za-z]{4,}\b', question)
            queries = [
                question,
                f"{' '.join(words[:3])} symptoms diagnosis" if words else question,
                f"{' '.join(words[:3])} treatment" if words else question,
                f"{' '.join(words[1:4])} medical" if len(words) > 1 else question,
                f"{' '.join(words[:2])} pathophysiology" if len(words) > 1 else question,
            ][:self.num_subqueries]
        
        return queries
    
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
                temperature=1.0, do_sample=False,
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
        
        # CoT prompt format from MIRAGE template
        user_prompt = f"""Here is the question:
{question}

Here are the potential choices:
{options_text}

Please think step-by-step and generate your output in json:"""
        
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
        
        user_prompt = f"""Here are the relevant documents:
{context}

Here is the question:
{question}

Here are the potential choices:
{options_text}

Please think step-by-step and generate your output in json:"""
        
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
        """Evaluate a single question across selected modes with maximum parallelism"""
        
        if modes is None:
            modes = ["direct", "baseline", "planning"]
        
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
            
            # Handle RAG modes (including planning_v2, planning_v3, planning_v4, planning_v5, planning_v6)
            rag_modes = [m for m in modes if m in ["direct", "baseline", "planning", "planning_v2", "planning_v3", "planning_v4", "planning_v4_grpo", "planning_v5", "planning_v6"]]
            
            if rag_modes:
                # Step 1: Generate plans and baseline queries in parallel
                tasks = []
                task_names = []
                
                need_plan = "planning" in rag_modes
                need_plan_v2 = "planning_v2" in rag_modes
                need_plan_v3 = "planning_v3" in rag_modes
                need_plan_v4 = "planning_v4" in rag_modes
                need_plan_v4_grpo = "planning_v4_grpo" in rag_modes
                need_plan_v5 = "planning_v5" in rag_modes
                need_plan_v6 = "planning_v6" in rag_modes
                need_baseline_queries = "baseline" in rag_modes
                
                if need_plan:
                    tasks.append(self.generate_plan_async(question, session))
                    task_names.append("plan")
                if need_plan_v2:
                    tasks.append(self.generate_plan_v2_async(question, options, session))
                    task_names.append("plan_v2")
                if need_plan_v3:
                    tasks.append(self.generate_plan_v3_async(question, options, session))
                    task_names.append("plan_v3")
                if need_plan_v4:
                    tasks.append(self.generate_plan_v4_async(question, options, session))
                    task_names.append("plan_v4")
                if need_plan_v4_grpo:
                    tasks.append(self.generate_plan_v4_async(question, options, session))
                    task_names.append("plan_v4_grpo")
                if need_plan_v5:
                    # V5 uses same plan as V4
                    tasks.append(self.generate_plan_v4_async(question, options, session))
                    task_names.append("plan_v5")
                if need_plan_v6:
                    tasks.append(self.generate_plan_v6_async(question, options, session))
                    task_names.append("plan_v6")
                if need_baseline_queries:
                    tasks.append(self.generate_subqueries_async(question, plan=None, session=session))
                    task_names.append("baseline_queries")
                
                gathered = await asyncio.gather(*tasks) if tasks else []
                
                plan = None
                plan_v2 = None
                plan_v3 = None
                plan_v4 = None
                plan_v4_grpo = None
                plan_v5 = None
                plan_v6 = None
                baseline_queries = None
                for i, name in enumerate(task_names):
                    if name == "plan":
                        plan = gathered[i]
                    elif name == "plan_v2":
                        plan_v2 = gathered[i]
                    elif name == "plan_v3":
                        plan_v3 = gathered[i]
                    elif name == "plan_v4":
                        plan_v4 = gathered[i]
                    elif name == "plan_v4_grpo":
                        plan_v4_grpo = gathered[i]
                    elif name == "plan_v5":
                        plan_v5 = gathered[i]
                    elif name == "plan_v6":
                        plan_v6 = gathered[i]
                    elif name == "baseline_queries":
                        baseline_queries = gathered[i]
                
                # Step 2: Generate queries based on plans
                planning_queries = None
                planning_v2_queries = None
                planning_v3_queries = None
                planning_v4_queries = None
                planning_v4_grpo_queries = None
                planning_v5_queries = None
                planning_v6_queries = None
                
                query_tasks = []
                query_task_names = []
                
                if "planning" in rag_modes and plan:
                    query_tasks.append(self.generate_subqueries_async(question, plan=plan, session=session))
                    query_task_names.append("planning_queries")
                if "planning_v2" in rag_modes and plan_v2:
                    query_tasks.append(self.generate_queries_from_plan_v2_async(question, plan_v2, session))
                    query_task_names.append("planning_v2_queries")
                if "planning_v3" in rag_modes and plan_v3:
                    query_tasks.append(self.generate_queries_from_plan_v3_async(question, plan_v3, session))
                    query_task_names.append("planning_v3_queries")
                if "planning_v4" in rag_modes and plan_v4:
                    query_tasks.append(self.generate_queries_from_plan_v4_async(question, plan_v4, session))
                    query_task_names.append("planning_v4_queries")
                if "planning_v4_grpo" in rag_modes and plan_v4_grpo:
                    # GRPO rewriter is sync (HF model), run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    planning_v4_grpo_queries = await loop.run_in_executor(
                        None, self.generate_queries_grpo_rewriter, question, options, plan_v4_grpo
                    )
                if "planning_v5" in rag_modes and plan_v5:
                    # V5 uses same query generation as V4
                    query_tasks.append(self.generate_queries_from_plan_v4_async(question, plan_v5, session))
                    query_task_names.append("planning_v5_queries")
                if "planning_v6" in rag_modes and plan_v6:
                    query_tasks.append(self.generate_queries_from_plan_v6_async(question, plan_v6, session))
                    query_task_names.append("planning_v6_queries")
                
                query_gathered = await asyncio.gather(*query_tasks) if query_tasks else []
                for i, name in enumerate(query_task_names):
                    if name == "planning_queries":
                        planning_queries = query_gathered[i]
                    elif name == "planning_v2_queries":
                        planning_v2_queries = query_gathered[i]
                    elif name == "planning_v3_queries":
                        planning_v3_queries = query_gathered[i]
                    elif name == "planning_v4_queries":
                        planning_v4_queries = query_gathered[i]
                    elif name == "planning_v4_grpo_queries":
                        planning_v4_grpo_queries = query_gathered[i]
                    elif name == "planning_v5_queries":
                        planning_v5_queries = query_gathered[i]
                    elif name == "planning_v6_queries":
                        planning_v6_queries = query_gathered[i]
                
                # Step 3: Retrieve documents for each mode
                direct_docs = []
                baseline_docs = []
                planning_docs = []
                planning_v2_docs = []
                planning_v3_docs = []
                planning_v4_docs = []
                planning_v4_grpo_docs = []
                planning_v5_docs = []
                planning_v6_docs = []
                
                # Each mode retrieves total_docs documents (budget split across queries)
                T = self.total_docs
                if "direct" in rag_modes:
                    direct_docs = self.retrieve_documents([question], T)
                if "baseline" in rag_modes and baseline_queries:
                    k = max(1, T // len(baseline_queries))
                    baseline_docs = self.retrieve_documents(baseline_queries, k)
                if "planning" in rag_modes and planning_queries:
                    k = max(1, T // len(planning_queries))
                    planning_docs = self.retrieve_documents(planning_queries, k)
                if "planning_v2" in rag_modes and planning_v2_queries:
                    k = max(1, T // len(planning_v2_queries))
                    planning_v2_docs = self.retrieve_documents(planning_v2_queries, k)
                if "planning_v3" in rag_modes and planning_v3_queries:
                    k = max(1, T // len(planning_v3_queries))
                    planning_v3_docs = self.retrieve_documents(planning_v3_queries, k)
                if "planning_v4" in rag_modes and planning_v4_queries:
                    k = max(1, T // len(planning_v4_queries))
                    planning_v4_docs = self.retrieve_documents(planning_v4_queries, k)
                if "planning_v4_grpo" in rag_modes and planning_v4_grpo_queries:
                    k = max(1, T // len(planning_v4_grpo_queries))
                    planning_v4_grpo_docs = self.retrieve_documents(planning_v4_grpo_queries, k)
                if "planning_v5" in rag_modes and planning_v5_queries:
                    k = max(1, T // len(planning_v5_queries))
                    planning_v5_docs = self.retrieve_documents(planning_v5_queries, k)
                if "planning_v6" in rag_modes and planning_v6_queries:
                    k = max(1, T // len(planning_v6_queries))
                    planning_v6_docs = self.retrieve_documents(planning_v6_queries, k)
                
                # Step 4: Generate all answers in parallel
                answer_tasks = []
                answer_mode_order = []
                
                if "direct" in rag_modes:
                    answer_tasks.append(self.generate_answer_async(question, options, direct_docs, session))
                    answer_mode_order.append("direct")
                if "baseline" in rag_modes and baseline_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, baseline_docs, session))
                    answer_mode_order.append("baseline")
                if "planning" in rag_modes and planning_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, planning_docs, session))
                    answer_mode_order.append("planning")
                if "planning_v2" in rag_modes and planning_v2_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, planning_v2_docs, session))
                    answer_mode_order.append("planning_v2")
                if "planning_v3" in rag_modes and planning_v3_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, planning_v3_docs, session))
                    answer_mode_order.append("planning_v3")
                if "planning_v4" in rag_modes and planning_v4_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, planning_v4_docs, session))
                    answer_mode_order.append("planning_v4")
                if "planning_v4_grpo" in rag_modes and planning_v4_grpo_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, planning_v4_grpo_docs, session))
                    answer_mode_order.append("planning_v4_grpo")
                if "planning_v5" in rag_modes and planning_v5_queries:
                    # V5 uses the new method that includes plan and subqueries
                    answer_tasks.append(self.generate_answer_with_plan_async(question, options, planning_v5_docs, plan_v5, planning_v5_queries, session))
                    answer_mode_order.append("planning_v5")
                if "planning_v6" in rag_modes and planning_v6_queries:
                    answer_tasks.append(self.generate_answer_async(question, options, planning_v6_docs, session))
                    answer_mode_order.append("planning_v6")
                
                answer_results = await asyncio.gather(*answer_tasks) if answer_tasks else []
                
                # Build results
                for i, mode in enumerate(answer_mode_order):
                    raw_resp, predicted = answer_results[i]
                    
                    if mode == "direct":
                        docs = direct_docs
                        queries = [question]
                        mode_plan = None
                    elif mode == "baseline":
                        docs = baseline_docs
                        queries = baseline_queries if baseline_queries else []
                        mode_plan = None
                    elif mode == "planning":
                        docs = planning_docs
                        queries = planning_queries if planning_queries else []
                        mode_plan = plan
                    elif mode == "planning_v2":
                        docs = planning_v2_docs
                        queries = planning_v2_queries if planning_v2_queries else []
                        mode_plan = plan_v2
                    elif mode == "planning_v3":
                        docs = planning_v3_docs
                        queries = planning_v3_queries if planning_v3_queries else []
                        mode_plan = plan_v3
                    elif mode == "planning_v4":
                        docs = planning_v4_docs
                        queries = planning_v4_queries if planning_v4_queries else []
                        mode_plan = plan_v4
                    elif mode == "planning_v5":
                        docs = planning_v5_docs
                        queries = planning_v5_queries if planning_v5_queries else []
                        mode_plan = plan_v5
                    elif mode == "planning_v4_grpo":
                        docs = planning_v4_grpo_docs
                        queries = planning_v4_grpo_queries if planning_v4_grpo_queries else []
                        mode_plan = plan_v4_grpo
                    elif mode == "planning_v6":
                        docs = planning_v6_docs
                        queries = planning_v6_queries if planning_v6_queries else []
                        mode_plan = plan_v6
                    
                    result["modes"][mode] = {
                        "num_queries": len(queries),
                        "num_docs": len(docs),
                        "queries": queries,
                        "plan": mode_plan,
                        "retrieved_docs": docs,  # Full documents with title, content, scores
                        "raw_response": raw_resp,
                        "predicted_answer": predicted,
                        "is_correct": predicted.upper() == correct_answer.upper(),
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
        """Evaluate a single question across all 3 RAG modes with maximum parallelism"""
        return await self.evaluate_question_selected_modes(
            question_data, question_id, session,
            modes=["direct", "baseline", "planning"]
        )


async def run_evaluation_async(
    max_questions: int = 100,
    llm_provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    max_concurrent: int = 100,
    output_dir: str = "results",
    modes: List[str] = None,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_tokens: int = 4096,
    vllm_max_concurrent: int = 1,
    vllm_max_model_len: int = 8192,
    total_docs: int = 15,
    rewriter_adapter_path: Optional[str] = None,
    rewriter_base_model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Run maximum performance async evaluation with selectable modes"""
    
    if modes is None:
        modes = ["direct", "baseline", "planning"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MedQA Evaluation (MAX ASYNC)")
    print("=" * 80)
    print(f"LLM provider: {llm_provider}")
    print(f"Model: {model_name}")
    print(f"Modes: {modes}")
    print(f"Max questions: {max_questions}")
    print(f"Max concurrent requests: {max_concurrent}")
    if llm_provider == "vllm":
        print(f"vLLM settings: tp={vllm_tensor_parallel_size}, gpu_mem={vllm_gpu_memory_utilization}, max_model_len={vllm_max_model_len}, max_tokens={vllm_max_tokens}")
    
    if "cot" in modes:
        print(f"CoT: No retrieval (Chain-of-Thought only)")
    
    rag_modes = [m for m in modes if m != "cot"]
    if rag_modes:
        print(f"RAG: total_docs=15 per mode (budget split across queries)")
        print(f"Retriever: {retriever_name} | Corpus: {corpus_name}")
    
    print("=" * 80)
    
    # Load dataset
    dataset = MedQADataset()
    total_questions = min(max_questions, len(dataset))
    
    # Initialize evaluator
    evaluator = AsyncRAGEvaluator(
        llm_provider=llm_provider,
        model_name=model_name,
        retriever_name=retriever_name,
        corpus_name=corpus_name,
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
        
        # Create all tasks at once
        tasks = []
        for i in range(total_questions):
            question_data = dataset[i]
            task = evaluator.evaluate_question_selected_modes(question_data, i, session, modes=modes)
            tasks.append(task)
        
        # Estimate time based on mode complexity
        api_calls_per_question = len([m for m in modes if m == "cot"])  # 1 for CoT
        if "direct" in modes:
            api_calls_per_question += 1
        if "baseline" in modes:
            api_calls_per_question += 2  # subquery gen + answer
        if "planning" in modes:
            api_calls_per_question += 3  # plan + subquery gen + answer
        
        est_minutes = total_questions * api_calls_per_question / (max_concurrent * 3)  # rough estimate
        
        # Run all tasks with progress bar
        print(f"\nProcessing {total_questions} questions with {max_concurrent} concurrent requests...")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Estimated API calls: ~{total_questions * api_calls_per_question}")
        print("=" * 80)
        
        completed_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
        
        # Process results with periodic logging
        for idx, result in enumerate(completed_results):
            if result is None or isinstance(result, Exception):
                continue
            
            results.append(result)
            
            for mode in modes:
                mode_result = result.get("modes", {}).get(mode, {})
                if "error" not in mode_result and "is_correct" in mode_result:
                    mode_stats[mode]["total"] += 1
                    if mode_result.get("is_correct", False):
                        mode_stats[mode]["correct"] += 1
            
            # Print progress every 50 questions
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta_seconds = (total_questions - idx - 1) / rate if rate > 0 else 0
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: {idx+1}/{total_questions} ({(idx+1)/total_questions*100:.1f}%)")
                print(f"  Elapsed: {elapsed/60:.1f} min | Speed: {rate*60:.1f} q/min | ETA: {eta_seconds/60:.1f} min")
                
                for mode in modes:
                    stats = mode_stats[mode]
                    if stats["total"] > 0:
                        acc = stats["correct"] / stats["total"] * 100
                        print(f"  {mode:12s}: {stats['correct']:3d}/{stats['total']:3d} ({acc:.1f}%)")
                print("-" * 80)
    
    total_time = time.time() - start_time
    
    # Build summary
    summary = {
        "config": {
            "llm_provider": llm_provider,
            "model_name": model_name,
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
    output_file = os.path.join(output_dir, f"medqa_{mode_suffix}_{model_suffix}_{timestamp}.json")
    
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


def _run_vllm_batch_phase(model_path: str, messages_list_file: str,
                          output_file: str, gpu_mem: float,
                          max_model_len: int, max_tokens: int):
    """
    Subprocess entry: load vLLM model, run batch chat, save outputs, exit.
    All GPU memory is freed when this process exits.
    """
    from vllm import LLM, SamplingParams

    # Load messages
    with open(messages_list_file, "rb") as f:
        messages_list = pickle.load(f)

    print(f"    [subprocess] Loading model: {os.path.basename(model_path)} ...")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
    )

    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
    print(f"    [subprocess] Running batch inference on {len(messages_list)} inputs ...")

    prompts = [_format_chat_messages(llm.get_tokenizer(), m) for m in messages_list]
    try:
        outputs = llm.chat(messages=messages_list, sampling_params=sampling)
    except Exception:
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)

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
                             max_tokens: int) -> list:
    """Run a vLLM batch in a subprocess and return result texts."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f_in:
        pickle.dump(messages_list, f_in)
        input_path = f_in.name

    output_path = input_path + ".out.pkl"

    try:
        p = _mp.Process(
            target=_run_vllm_batch_phase,
            args=(model_path, input_path, output_path,
                  gpu_mem, max_model_len, max_tokens),
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
    max_tokens: int = 2048,
    max_concurrent: int = 100,
    output_dir: str = "outputs",
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    hypothesis_prompt: str = "v1",
    rewriting_prompt: str = "v1",
    generator_prompt: str = "v1",
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Batch-phased hypothesis evaluation via OpenAI API (GPT-compatible).

    This mirrors run_batch_phased_evaluation() but uses async API calls
    instead of vLLM subprocess inference.
    """
    from training.reward import (
        parse_queries_from_completion, parse_hypothesis_plan,
    )

    api_key = resolve_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Set env or add export to ~/.bashrc.")

    # Resolve per-module models
    hypothesis_model = hypothesis_checkpoint or base_model
    rewriter_model = rewriter_checkpoint or base_model
    generator_model = generator_checkpoint or base_model

    h_prompt = HYPOTHESIS_PROMPTS[hypothesis_prompt]
    r_prompt = REWRITING_PROMPTS[rewriting_prompt]
    g_prompt = GENERATOR_PROMPTS[generator_prompt]
    combo_tag = f"h{hypothesis_prompt}_r{rewriting_prompt}_g{generator_prompt}"

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

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset = MedQADataset()
    n = min(max_questions, len(dataset))
    questions = [dataset[i] for i in range(n)]
    print(f"\n{'='*70}")
    print(f"Batch-Phased Evaluation (OpenAI): {n} questions")
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
    if api_base:
        print(f"api_base={api_base}")
    print(f"{'='*70}")

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
            if not p["discriminating_features"]:
                words = re.findall(r'\b[A-Za-z]{4,}\b', questions[idx]["question"])
                p["discriminating_features"] = words[:3] or ["symptom"]
            plans.append(p)

        print(f"  ✓ Phase 1 done: {len(plans)} hypotheses "
              f"in {time.time()-t1:.1f}s")

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
            alt_letter = (p.get("alternative_if_wrong", "") or "").strip().upper().rstrip(".")
            alt_text = (
                f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}"
                if alt_letter in qd.get("options", {}) else alt_letter
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
            from retriever import create_retriever as _cr
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
            }
            if generator_prompt == "v2":
                p = plans[i]
                fmt_vars["hypothesis_summary"] = (
                    f"Best guess: {p.get('best_guess', '')} — {p.get('reasoning', '')}"
                )
                fmt_vars["queries_summary"] = "\n".join(
                    f"  {j+1}. {q}" for j, q in enumerate(all_queries[i])
                )

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
            "question_id": i,
            "question": qd["question"],
            "options": qd["options"],
            "correct_answer": correct_answer,
            "modes": {
                combo_tag: {
                    "num_queries": len(all_queries[i]),
                    "num_docs": len(all_docs[i]),
                    "queries": all_queries[i],
                    "plan": plans[i],
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
            "llm_provider": "openai",
            "base_model": base_model,
            "hypothesis_checkpoint": hypothesis_checkpoint,
            "rewriter_checkpoint": rewriter_checkpoint,
            "generator_checkpoint": generator_checkpoint,
            "hypothesis_prompt": hypothesis_prompt,
            "rewriting_prompt": rewriting_prompt,
            "generator_prompt": generator_prompt,
            "combo_tag": combo_tag,
            "total_evaluated": total,
            "total_docs": total_docs,
            "max_tokens": max_tokens,
            "max_concurrent": max_concurrent,
            "api_base": api_base,
        },
        "timing": {
            "total_seconds": elapsed,
            "avg_per_question": elapsed / total if total else 0,
            "questions_per_minute": total / elapsed * 60 if elapsed > 0 else 0,
        },
        "mode_results": {
            combo_tag: {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        },
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\n{'='*70}")
    print(f"[{combo_tag}] {accuracy:.2f}% ({correct}/{total}) "
          f"in {elapsed:.0f}s ({summary['timing']['questions_per_minute']:.1f} q/min)")
    print(f"{'='*70}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", base_model).strip("_")
    if not model_suffix:
        model_suffix = "model"
    out_file = os.path.join(
        output_dir,
        f"medqa_{combo_tag}_{model_suffix}_{ts}.json",
    )
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {out_file}")

    return {
        "combo": combo_tag,
        "accuracy": accuracy,
        "correct": correct,
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
    gpu_mem: float = 0.9,
    max_model_len: int = 8192,
    max_tokens: int = 2048,
    output_dir: str = "outputs",
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    hypothesis_prompt: str = "v1",
    rewriting_prompt: str = "v1",
    generator_prompt: str = "v1",
) -> Dict[str, Any]:
    """
    Batch-phased evaluation with selectable prompts.
    Each vLLM phase runs in a separate subprocess → full GPU memory release.

    Args:
        hypothesis_prompt: "v1", "v2", or "v3" (from prompts.py)
        rewriting_prompt:  "v1", "v2", or "v3" (from prompts.py)
        generator_prompt:  "v1" or "v2"        (from prompts.py)
    """
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
    dataset = MedQADataset()
    n = min(max_questions, len(dataset))
    questions = [dataset[i] for i in range(n)]
    print(f"\n{'='*70}")
    print(f"Batch-Phased Evaluation: {n} questions")
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
    print(f"gpu_mem={gpu_mem}  max_model_len={max_model_len}  "
          f"max_tokens={max_tokens}")
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
        hypothesis_model, plan_messages, gpu_mem, max_model_len, max_tokens)

    plans = []
    for idx, text in enumerate(plan_texts):
        p = parse_hypothesis_plan(text)
        p.setdefault("discriminating_features", [])
        p.setdefault("best_guess", "")
        p.setdefault("reasoning", "")
        p.setdefault("confirming_evidence", [])
        p.setdefault("alternative_if_wrong", "")
        if not p["discriminating_features"]:
            words = re.findall(
                r'\b[A-Za-z]{4,}\b', questions[idx]["question"])
            p["discriminating_features"] = words[:3] or ["symptom"]
        plans.append(p)

    print(f"  ✓ Phase 1 done: {len(plans)} hypotheses "
          f"in {time.time()-t1:.1f}s")

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
        alt_letter = (p.get("alternative_if_wrong", "") or "").strip().upper().rstrip(".")
        alt_text = f"{alt_letter}. {qd['options'].get(alt_letter, alt_letter)}" if alt_letter in qd.get('options', {}) else alt_letter
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
        )
        rw_messages.append([
            {"role": "system", "content": r_prompt["system"]},
            {"role": "user", "content": user},
        ])

    rw_texts = _run_phase_in_subprocess(
        rewriter_model, rw_messages, gpu_mem, max_model_len, max_tokens)

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
        from retriever import create_retriever as _cr
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
        }
        # generator v2 needs extra context
        if generator_prompt == "v2":
            p = plans[i]
            fmt_vars["hypothesis_summary"] = (
                f"Best guess: {p.get('best_guess','')} — "
                f"{p.get('reasoning','')}"
            )
            fmt_vars["queries_summary"] = "\n".join(
                f"  {j+1}. {q}" for j, q in enumerate(all_queries[i]))

        user = g_prompt["user"].format(**fmt_vars)
        ans_messages.append([
            {"role": "system", "content": g_prompt["system"]},
            {"role": "user", "content": user},
        ])

    ans_texts = _run_phase_in_subprocess(
        generator_model, ans_messages, gpu_mem, max_model_len, max_tokens)

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
            "question_id": i,
            "question": qd["question"],
            "options": qd["options"],
            "correct_answer": correct_answer,
            "modes": {
                combo_tag: {
                    "num_queries": len(all_queries[i]),
                    "num_docs": len(all_docs[i]),
                    "queries": all_queries[i],
                    "plan": plans[i],
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
            "combo_tag": combo_tag,
            "total_evaluated": total,
            "total_docs": total_docs,
            "gpu_mem": gpu_mem,
            "max_model_len": max_model_len,
            "max_tokens": max_tokens,
        },
        "timing": {
            "total_seconds": elapsed,
            "avg_per_question": elapsed / total if total else 0,
            "questions_per_minute":
                total / elapsed * 60 if elapsed > 0 else 0,
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
    print(f"{'='*70}")

    # Save — name encodes prompt combination
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"medqa_{combo_tag}_{ts}.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": results},
                  f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {out_file}")

    return {"combo": combo_tag, "accuracy": accuracy,
            "correct": correct, "total": total, "file": out_file}


def main():
    parser = argparse.ArgumentParser(
        description='MedQA RAG Evaluation v2 (Prompt-Selectable Ablation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Hypothesis mode with specific prompts
  python evaluate_medqa_v2.py --mode hypothesis \\
      --hypothesis-prompt v1 --rewriting-prompt v2 --generator-prompt v1 \\
      --model Qwen/Qwen3-4B-Instruct-2507 --max-questions 1273

  # Run all 9 hypothesis×rewriting combos (generator=v1)
  python evaluate_medqa_v2.py --mode hypothesis --run-all \\
      --model Qwen/Qwen3-4B-Instruct-2507 --max-questions 1273

  # With checkpoints
  python evaluate_medqa_v2.py --mode hypothesis \\
      --hypothesis-prompt v1 --rewriting-prompt v1 \\
      --model Qwen/Qwen3-4B-Instruct-2507 \\
      --hypothesis-checkpoint /path/to/checkpoint-1100

  # Baseline modes (cot, direct, baseline)
  python evaluate_medqa_v2.py --mode cot --max-questions 1273
  python evaluate_medqa_v2.py --mode direct --max-questions 1273
  python evaluate_medqa_v2.py --mode baseline --max-questions 1273
        """
    )

    # ── Mode selection ──
    parser.add_argument(
        '--mode', type=str, default='hypothesis',
        choices=['cot', 'direct', 'baseline', 'hypothesis'],
        help='Evaluation mode (default: hypothesis)')
    parser.add_argument(
        '--hypothesis-prompt', type=str, default='v1',
        choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
        help='Hypothesis prompt version (default: v1)')
    parser.add_argument(
        '--rewriting-prompt', type=str, default='v1',
        choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'],
        help='Rewriting prompt version (default: v1)')
    parser.add_argument(
        '--generator-prompt', type=str, default='v1',
        choices=['v1', 'v2'],
        help='Generator prompt version (default: v1)')
    parser.add_argument(
        '--run-all', action='store_true',
        help='Run all 9 hypothesis×rewriting combos with generator=v1')

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
    parser.add_argument('--corpus', type=str, default=None,
                       help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', '-o', type=str, default='outputs')
    parser.add_argument('--api-base', type=str, default=None)
    parser.add_argument('--total-docs', type=int, default=15)

    # ── vLLM settings ──
    parser.add_argument('--vllm-tensor-parallel-size', type=int, default=1)
    parser.add_argument('--vllm-gpu-memory-utilization', type=float,
                       default=0.9)
    parser.add_argument('--vllm-max-tokens', type=int, default=2048)
    parser.add_argument('--vllm-max-concurrent', type=int, default=1)
    parser.add_argument('--vllm-max-model-len', type=int, default=8192)

    args = parser.parse_args()
    corpus_name = args.corpus if args.corpus else args.retrieval_dataset

    # ── Hypothesis mode (batch-phased) ──
    if args.mode == 'hypothesis':
        if args.run_all:
            # Run all 9 combos: hypothesis × rewriting, generator=v1
            combos = [
                (hv, rv)
                for hv in ['v1', 'v2', 'v3']
                for rv in ['v1', 'v2', 'v3']
            ]
            summary_rows = []
            print(f"\\n{'#'*70}")
            print(f"Running ALL {len(combos)} prompt combinations")
            print(f"{'#'*70}")
            for idx, (hv, rv) in enumerate(combos, 1):
                print(f"\\n\\n{'#'*70}")
                print(f"  Combo {idx}/{len(combos)}: "
                      f"hypothesis={hv}, rewriting={rv}, generator=v1")
                print(f"{'#'*70}")
                if args.llm_provider == "openai":
                    result = asyncio.run(run_batch_phased_evaluation_openai(
                        base_model=args.model,
                        rewriter_checkpoint=args.rewriter_checkpoint,
                        hypothesis_checkpoint=args.hypothesis_checkpoint,
                        max_questions=args.max_questions,
                        total_docs=args.total_docs,
                        max_tokens=args.vllm_max_tokens,
                        max_concurrent=args.max_concurrent,
                        output_dir=args.output_dir,
                        retriever_name=args.retriever,
                        corpus_name=corpus_name,
                        hypothesis_prompt=hv,
                        rewriting_prompt=rv,
                        generator_prompt='v1',
                        api_base=args.api_base,
                    ))
                else:
                    result = run_batch_phased_evaluation(
                        base_model=args.model,
                        rewriter_checkpoint=args.rewriter_checkpoint,
                        hypothesis_checkpoint=args.hypothesis_checkpoint,
                        max_questions=args.max_questions,
                        total_docs=args.total_docs,
                        gpu_mem=args.vllm_gpu_memory_utilization,
                        max_model_len=args.vllm_max_model_len,
                        max_tokens=args.vllm_max_tokens,
                        output_dir=args.output_dir,
                        retriever_name=args.retriever,
                        corpus_name=corpus_name,
                        hypothesis_prompt=hv,
                        rewriting_prompt=rv,
                        generator_prompt='v1',
                    )
                summary_rows.append(result)

            # Print summary table
            print(f"\\n\\n{'='*70}")
            print("ABLATION SUMMARY")
            print(f"{'='*70}")
            print(f"{'Combo':<15} {'Accuracy':>10} {'Correct':>10} "
                  f"{'Total':>8}")
            print(f"{'-'*15} {'-'*10} {'-'*10} {'-'*8}")
            for row in summary_rows:
                print(f"{row['combo']:<15} "
                      f"{row['accuracy']:>9.2f}% "
                      f"{row['correct']:>10} "
                      f"{row['total']:>8}")
            print(f"{'='*70}")

            # Save summary CSV
            import csv
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(
                args.output_dir, f"ablation_summary_{ts}.csv")
            with open(csv_file, "w", newline="") as f:
                w = csv.DictWriter(
                    f, fieldnames=["combo", "accuracy", "correct",
                                   "total", "file"])
                w.writeheader()
                w.writerows(summary_rows)
            print(f"✓ Summary CSV saved: {csv_file}")

        else:
            # Single hypothesis combo
            if args.llm_provider == "openai":
                asyncio.run(run_batch_phased_evaluation_openai(
                    base_model=args.model,
                    rewriter_checkpoint=args.rewriter_checkpoint,
                    hypothesis_checkpoint=args.hypothesis_checkpoint,
                    max_questions=args.max_questions,
                    total_docs=args.total_docs,
                    max_tokens=args.vllm_max_tokens,
                    max_concurrent=args.max_concurrent,
                    output_dir=args.output_dir,
                    retriever_name=args.retriever,
                    corpus_name=corpus_name,
                    hypothesis_prompt=args.hypothesis_prompt,
                    rewriting_prompt=args.rewriting_prompt,
                    generator_prompt=args.generator_prompt,
                    api_base=args.api_base,
                ))
            else:
                run_batch_phased_evaluation(
                    base_model=args.model,
                    rewriter_checkpoint=args.rewriter_checkpoint,
                    hypothesis_checkpoint=args.hypothesis_checkpoint,
                    max_questions=args.max_questions,
                    total_docs=args.total_docs,
                    gpu_mem=args.vllm_gpu_memory_utilization,
                    max_model_len=args.vllm_max_model_len,
                    max_tokens=args.vllm_max_tokens,
                    output_dir=args.output_dir,
                    retriever_name=args.retriever,
                    corpus_name=corpus_name,
                    hypothesis_prompt=args.hypothesis_prompt,
                    rewriting_prompt=args.rewriting_prompt,
                    generator_prompt=args.generator_prompt,
                )

    else:
        # ── Baseline modes (cot, direct, baseline) via async evaluator ──
        asyncio.run(run_evaluation_async(
            max_questions=args.max_questions,
            llm_provider=args.llm_provider,
            model_name=args.model,
            retriever_name=args.retriever,
            corpus_name=corpus_name,
            max_concurrent=args.max_concurrent,
            output_dir=args.output_dir,
            modes=[args.mode],
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_tokens=args.vllm_max_tokens,
            vllm_max_concurrent=args.vllm_max_concurrent,
            vllm_max_model_len=args.vllm_max_model_len,
            total_docs=args.total_docs,
            rewriter_adapter_path=args.rewriter_checkpoint,
            rewriter_base_model=args.rewriter_base_model,
            api_base=args.api_base,
        ))


if __name__ == "__main__":
    import multiprocessing as _main_mp
    _main_mp.set_start_method("spawn", force=True)
    main()
