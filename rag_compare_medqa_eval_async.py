#!/usr/bin/env python3
"""
MedQA Evaluation for RAG Comparison Experiment (Async Parallel Version)

This script evaluates different RAG modes on MedQA testset
with maximum async parallel execution for fastest processing.

Evaluation Modes:
- cot: Chain-of-Thought baseline (no RAG, just LLM reasoning)
- direct: 1 query × 25 documents = 25 documents total
- baseline: 5 queries × 5 documents = 25 documents total  
- planning: 5 queries × 5 documents = 25 documents total
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
sys.path.insert(0, os.path.dirname(__file__))

from retriever import create_retriever

# ============================================================================
# Prompts from MIRAGE template.py
# ============================================================================
# CoT (Chain-of-Thought) - No RAG
COT_SYSTEM_PROMPT = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

# RAG (with documents)
MIRAGE_SYSTEM_PROMPT = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE', 'MedRAG', 'src'))
    from template import general_medrag_system, general_cot_system
    MIRAGE_SYSTEM_PROMPT = general_medrag_system
    COT_SYSTEM_PROMPT = general_cot_system
    print("✓ Loaded MIRAGE prompts from template.py")
except ImportError:
    print("⚠ Using built-in MIRAGE prompts")


# ============================================================================
# Multi-Query Rewriting Prompt (5 queries version)
# ============================================================================
MULTI_QUERY_PROMPT_5 = """You are an AI language model assistant. Your task
is to generate exactly five different versions of the
given user question to retrieve relevant documents
from a vector database. By generating multiple
perspectives on the user question, your goal is to
help the user overcome some of the limitations of
the distance-based similarity search.
Original question: {query}
Format your response in plain text as:
Sub-query 1:
Sub-query 2:
Sub-query 3:
Sub-query 4:
Sub-query 5:"""

PLANNING_PROMPT = """Analyze the following medical question and extract key information.

Question: {question}

Extract:
1. observed_features: List 3-7 key symptoms, findings, or conditions mentioned
2. must_check_cooccurrence: List pairs of features whose co-occurrence is important
3. need_disambiguation: List any confusing concepts that need distinction

Output in JSON format:
{{"observed_features": [...], "must_check_cooccurrence": [[...]], "need_disambiguation": [...]}}"""

# ============================================================================
# IMPROVED Planning V2 Prompts
# ============================================================================
PLANNING_V2_PROMPT = """You are a medical expert. Analyze this clinical question and provide a diagnostic reasoning plan.

Question: {question}

Options:
{options}

Provide your analysis in JSON format:
{{
    "key_clinical_features": ["list 3-5 most important clinical clues from the question"],
    "primary_diagnosis_hypothesis": "most likely diagnosis based on the features",
    "differential_diagnoses": ["2-3 alternative diagnoses to consider"],
    "distinguishing_features": ["specific findings that would differentiate between diagnoses"],
    "search_strategy": "brief explanation of what information would be most helpful to find"
}}"""

PLANNING_V2_QUERY_PROMPT = """Based on the clinical analysis, generate 5 specific medical search queries to find evidence.

Clinical Question: {question}

Analysis:
- Key Features: {key_features}
- Primary Hypothesis: {primary_diagnosis}
- Differential Diagnoses: {differentials}
- Distinguishing Features: {distinguishing}

Generate 5 targeted queries that will help differentiate between the diagnoses:
Query 1: Search for the PRIMARY diagnosis with key symptoms
Query 2: Search for DIFFERENTIAL diagnosis comparison
Query 3: Search for DISTINGUISHING lab/imaging/clinical findings
Query 4: Search for PATHOPHYSIOLOGY of the most likely condition
Query 5: Search for TREATMENT or MANAGEMENT approach

Format:
Query 1: [your query]
Query 2: [your query]
Query 3: [your query]
Query 4: [your query]
Query 5: [your query]"""

# ============================================================================
# IMPROVED Planning V3 Prompts - Adaptive Query Generation
# ============================================================================
PLANNING_V3_PROMPT = """You are an expert medical diagnostician. Analyze this clinical question and create a focused diagnostic plan.

Question: {question}

Options:
{options}

Provide your analysis in JSON format:
{{
    "question_type": "factual|diagnostic|mechanism|treatment|comparison",
    "complexity": "simple|moderate|complex",
    "key_clinical_clues": ["list 2-4 most critical clinical findings"],
    "most_likely_answer": "A/B/C/D with brief reasoning",
    "confidence": "high|medium|low",
    "what_evidence_needed": ["list specific information that would confirm the answer"],
    "differential_if_uncertain": ["only if confidence is low, list alternatives to consider"]
}}

Be concise and focused. Identify the most discriminating features."""

PLANNING_V3_QUERY_PROMPT = """You are a medical search expert. Generate the MINIMUM number of highly specific search queries needed to answer this question.

Question: {question}

Analysis:
- Question Type: {question_type}
- Complexity: {complexity}
- Key Clues: {key_clues}
- Most Likely Answer: {likely_answer}
- Evidence Needed: {evidence_needed}
{differential_section}

IMPORTANT RULES:
1. Generate ONLY the queries that are truly necessary (minimum 2, maximum 7)
2. Each query must be highly specific and targeted
3. For simple/factual questions, 2-3 queries are enough
4. For complex differential diagnosis, use 4-6 queries
5. DO NOT generate redundant or overlapping queries

Generate your queries (only as many as needed):
Query 1: [essential query for the most likely answer]
Query 2: [query to verify/distinguish]
... (add more only if necessary)"""

# ============================================================================
# Planning V4: Answer-Focused Approach with Evidence Verification
# ============================================================================
PLANNING_V4_PROMPT = """You are an expert medical diagnostician taking a medical licensing exam.

Question: {question}

Options:
{options}

Step 1: Identify the KEY DISCRIMINATING FEATURES that distinguish between the options.
Step 2: Make your BEST GUESS for the answer based on medical knowledge.
Step 3: Identify what SPECIFIC EVIDENCE would CONFIRM your answer.

Output in JSON:
{{
    "discriminating_features": ["2-3 features that distinguish between options"],
    "best_guess": "A/B/C/D",
    "reasoning": "brief explanation why this is the best answer",
    "confirming_evidence": ["1-3 specific facts that would confirm this answer"],
    "alternative_if_wrong": "A/B/C/D - only if uncertain"
}}"""

PLANNING_V4_QUERY_PROMPT = """Generate 3 highly targeted search queries to find evidence for this medical question.

Question: {question}
Best Guess Answer: {best_guess}
Reasoning: {reasoning}
Evidence Needed: {confirming_evidence}
Key Features: {discriminating_features}

Generate 3 SPECIFIC queries:
Query 1: Find evidence supporting {best_guess} - focus on the main reasoning
Query 2: Find distinguishing criteria for differential diagnosis  
Query 3: Find specific clinical/pathological features

Format:
Query 1: [query]
Query 2: [query]
Query 3: [query]"""

# ============================================================================
# Planning V6: Dual Hypothesis Testing Approach
# ============================================================================
PLANNING_V6_PROMPT = """You are an expert medical diagnostician. Analyze this question using differential diagnosis approach.

Question: {question}

Options:
{options}

Generate TWO most likely diagnostic hypotheses and plan how to verify each:

Output in JSON:
{{
    "hypothesis_1": {{
        "diagnosis": "Most likely answer (A/B/C/D)",
        "key_reasoning": "Why this is most likely",
        "supporting_features": ["2-3 clinical features that support this"],
        "evidence_to_verify": ["Specific evidence that would confirm this hypothesis"]
    }},
    "hypothesis_2": {{
        "diagnosis": "Second most likely answer (A/B/C/D)",
        "key_reasoning": "Why this is also possible",
        "supporting_features": ["2-3 clinical features that support this"],
        "evidence_to_verify": ["Specific evidence that would confirm this hypothesis"]
    }},
    "discriminating_criteria": ["Key findings that distinguish between hypothesis 1 and 2"]
}}"""

PLANNING_V6_QUERY_PROMPT = """Generate 5-6 targeted search queries to verify both hypotheses.

Question: {question}

Hypothesis 1: {h1_diagnosis}
- Reasoning: {h1_reasoning}
- Evidence needed: {h1_evidence}

Hypothesis 2: {h2_diagnosis}
- Reasoning: {h2_reasoning}
- Evidence needed: {h2_evidence}

Discriminating criteria: {discriminating}

Generate 5-6 queries:
Query 1: Evidence for Hypothesis 1
Query 2: Evidence for Hypothesis 2
Query 3: Distinguishing features between H1 and H2
Query 4: Confirming pathophysiology/mechanism
Query 5: Clinical presentation comparison
Query 6 (optional): Treatment/management differences

Format:
Query 1: [query]
Query 2: [query]
Query 3: [query]
Query 4: [query]
Query 5: [query]
Query 6: [query if needed]"""""


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
    ):
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
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
        }
        
        async with self.semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
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
# Async RAG Evaluator
# ============================================================================
class AsyncRAGEvaluator:
    """Maximum performance async RAG evaluation"""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        api_key: Optional[str] = None,
        max_concurrent: int = 100,
        num_subqueries: int = 5,
        docs_per_query_direct: int = 25,
        docs_per_query_multi: int = 5,
    ):
        self.model_name = model_name
        self.num_subqueries = num_subqueries
        self.docs_per_query_direct = docs_per_query_direct
        self.docs_per_query_multi = docs_per_query_multi
        self.max_concurrent = max_concurrent
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Async OpenAI client
        self.client = AsyncOpenAIClient(
            api_key=self.api_key,
            model=model_name,
            max_concurrent=max_concurrent,
        )
        
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
            cooccurrence_str = " & ".join([f"({a}&{b})" for a, b in cooccurrence[:2]]) if cooccurrence else ""
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
            rag_modes = [m for m in modes if m in ["direct", "baseline", "planning", "planning_v2", "planning_v3", "planning_v4", "planning_v5", "planning_v6"]]
            
            if rag_modes:
                # Step 1: Generate plans and baseline queries in parallel
                tasks = []
                task_names = []
                
                need_plan = "planning" in rag_modes
                need_plan_v2 = "planning_v2" in rag_modes
                need_plan_v3 = "planning_v3" in rag_modes
                need_plan_v4 = "planning_v4" in rag_modes
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
                planning_v5_docs = []
                planning_v6_docs = []
                
                if "direct" in rag_modes:
                    direct_docs = self.retrieve_documents([question], self.docs_per_query_direct)
                if "baseline" in rag_modes and baseline_queries:
                    baseline_docs = self.retrieve_documents(baseline_queries, self.docs_per_query_multi)
                if "planning" in rag_modes and planning_queries:
                    planning_docs = self.retrieve_documents(planning_queries, self.docs_per_query_multi)
                if "planning_v2" in rag_modes and planning_v2_queries:
                    planning_v2_docs = self.retrieve_documents(planning_v2_queries, self.docs_per_query_multi)
                if "planning_v3" in rag_modes and planning_v3_queries:
                    planning_v3_docs = self.retrieve_documents(planning_v3_queries, self.docs_per_query_multi)
                if "planning_v4" in rag_modes and planning_v4_queries:
                    planning_v4_docs = self.retrieve_documents(planning_v4_queries, self.docs_per_query_multi)
                if "planning_v5" in rag_modes and planning_v5_queries:
                    planning_v5_docs = self.retrieve_documents(planning_v5_queries, self.docs_per_query_multi)
                if "planning_v6" in rag_modes and planning_v6_queries:
                    planning_v6_docs = self.retrieve_documents(planning_v6_queries, self.docs_per_query_multi)
                
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
    model_name: str = "gpt-4o-mini",
    retriever_name: str = "MedCPT",
    corpus_name: str = "Textbooks",
    max_concurrent: int = 100,
    output_dir: str = "results",
    modes: List[str] = None,
) -> Dict[str, Any]:
    """Run maximum performance async evaluation with selectable modes"""
    
    if modes is None:
        modes = ["direct", "baseline", "planning"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("MedQA Evaluation (MAX ASYNC)")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Modes: {modes}")
    print(f"Max questions: {max_questions}")
    print(f"Max concurrent requests: {max_concurrent}")
    
    if "cot" in modes:
        print(f"CoT: No retrieval (Chain-of-Thought only)")
    
    rag_modes = [m for m in modes if m != "cot"]
    if rag_modes:
        print(f"RAG: direct=25 docs, baseline/planning=5x5 docs")
        print(f"Retriever: {retriever_name} | Corpus: {corpus_name}")
    
    print("=" * 80)
    
    # Load dataset
    dataset = MedQADataset()
    total_questions = min(max_questions, len(dataset))
    
    # Initialize evaluator
    evaluator = AsyncRAGEvaluator(
        model_name=model_name,
        retriever_name=retriever_name,
        corpus_name=corpus_name,
        max_concurrent=max_concurrent,
    )
    
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
            "model_name": model_name,
            "modes": modes,
            "retriever_name": retriever_name if rag_modes else None,
            "corpus_name": corpus_name if rag_modes else None,
            "max_questions": max_questions,
            "total_evaluated": len(results),
            "max_concurrent": max_concurrent,
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
    output_file = os.path.join(output_dir, f"medqa_{mode_suffix}_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="MedQA Max Async Parallel Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CoT baseline only (no RAG)
  python rag_compare_medqa_eval_async.py --modes cot --max-questions 1273

  # RAG modes only
  python rag_compare_medqa_eval_async.py --modes direct baseline planning --max-questions 1273

  # All modes including CoT
  python rag_compare_medqa_eval_async.py --modes cot direct baseline planning --max-questions 100

  # Quick test (10 questions)
  python rag_compare_medqa_eval_async.py --max-questions 10

  # Full RAG evaluation with max parallelism
  python rag_compare_medqa_eval_async.py --max-questions 1273 --max-concurrent 100
        """
    )
    
    parser.add_argument('--max-questions', '-n', type=int, default=2000)
    parser.add_argument('--max-concurrent', '-c', type=int, default=100,
                       help='Max concurrent API requests (default: 100)')
    parser.add_argument('--modes', '-m', nargs='+', 
                       choices=['cot', 'direct', 'baseline', 'planning', 'planning_v2', 'planning_v3', 'planning_v4', 'planning_v5', 'planning_v6'],
                       default=['direct', 'baseline', 'planning'],
                       help='Modes: v2=diagnostic, v3=adaptive, v4=answer-focused, v5=plan+queries, v6=dual-hypothesis')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--retriever', type=str, default='MedCPT')
    parser.add_argument('--corpus', type=str, default='Textbooks')
    parser.add_argument('--output-dir', '-o', type=str, default='results')
    
    args = parser.parse_args()
    
    # Run async evaluation
    asyncio.run(run_evaluation_async(
        max_questions=args.max_questions,
        model_name=args.model,
        retriever_name=args.retriever,
        corpus_name=args.corpus,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        modes=args.modes,
    ))


if __name__ == "__main__":
    main()
