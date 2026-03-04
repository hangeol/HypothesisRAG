#!/usr/bin/env python3
"""
Multi-Subquery RAG System using LangGraph

This system:
1. Generates multiple subqueries from an original question
2. Retrieves documents for each subquery
3. Generates responses for each subquery based on retrieved documents
4. Combines all responses to generate a final answer
"""

import os
import sys
import json
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import VLLM

# Add MIRAGE paths for retrieval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE', 'MedRAG'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'MIRAGE', 'MedRAG', 'src'))

try:
    from MedRAG.src.utils import RetrievalSystem
    RETRIEVAL_AVAILABLE = True
except ImportError:
    print("Warning: MIRAGE retrieval system not available. Using mock retrieval.")
    RETRIEVAL_AVAILABLE = False


class SubqueryResult(TypedDict):
    """Result from processing a single subquery"""
    subquery: str
    documents: List[Dict[str, Any]]
    response: str
    scores: List[float]


class GraphState(TypedDict):
    """State for the multi-subquery RAG graph"""
    # Input
    original_question: str
    options: Optional[Dict[str, str]]
    
    # Subquery generation
    subqueries: List[str]
    
    # Per-subquery results (accumulated)
    subquery_results: Annotated[List[SubqueryResult], add]
    
    # Final output
    final_answer: str
    
    # Metadata
    num_subqueries: int
    k_per_subquery: int


class MultiSubqueryRAG:
    """
    LangGraph-based Multi-Subquery RAG System
    
    This system orchestrates:
    1. Subquery generation from the original question
    2. Parallel document retrieval for each subquery
    3. Response generation for each subquery
    4. Final answer synthesis from all subquery responses
    """
    
    def __init__(
        self,
        llm_provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        retriever_name: str = "MedCPT",
        corpus_name: str = "Textbooks",
        db_dir: str = None,
        num_subqueries: int = 3,
        k_per_subquery: int = 5,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Multi-Subquery RAG system.
        
        Args:
            llm_provider: LLM provider ("openai", "anthropic", "vllm")
            model_name: Name of the model to use
            retriever_name: Name of the retriever (e.g., "MedCPT", "BM25")
            corpus_name: Name of the corpus (e.g., "Textbooks", "PubMed")
            db_dir: Directory for corpus data
            num_subqueries: Number of subqueries to generate
            k_per_subquery: Number of documents to retrieve per subquery
            temperature: Temperature for LLM generation
            api_key: API key for the LLM provider
        """
        self.num_subqueries = num_subqueries
        self.k_per_subquery = k_per_subquery
        
        # Initialize LLM
        self.llm = self._init_llm(llm_provider, model_name, temperature, api_key)
        
        # Initialize retrieval system
        if db_dir is None:
            db_dir = os.path.join(os.path.dirname(__file__), '..', 'MIRAGE', 'MedRAG', 'corpus')
        
        if RETRIEVAL_AVAILABLE:
            try:
                self.retrieval_system = RetrievalSystem(
                    retriever_name=retriever_name,
                    corpus_name=corpus_name,
                    db_dir=db_dir,
                    cache=True
                )
                print(f"✓ Retrieval system initialized with {retriever_name} on {corpus_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize retrieval system: {e}")
                self.retrieval_system = None
        else:
            self.retrieval_system = None
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _init_llm(self, provider: str, model_name: str, temperature: float, api_key: Optional[str]):
        """Initialize the LLM based on provider"""
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
        elif provider == "vllm":
            # For local models using vLLM
            return VLLM(
                model=model_name,
                temperature=temperature,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("generate_subqueries", self._generate_subqueries)
        workflow.add_node("process_subqueries", self._process_subqueries)
        workflow.add_node("synthesize_answer", self._synthesize_answer)
        
        # Define edges
        workflow.add_edge(START, "generate_subqueries")
        workflow.add_edge("generate_subqueries", "process_subqueries")
        workflow.add_edge("process_subqueries", "synthesize_answer")
        workflow.add_edge("synthesize_answer", END)
        
        # Compile
        return workflow.compile()
    
    def _generate_subqueries(self, state: GraphState) -> Dict[str, Any]:
        """Generate multiple subqueries from the original question"""
        question = state["original_question"]
        options = state.get("options", None)
        num_subqueries = state.get("num_subqueries", self.num_subqueries)
        
        # Build the prompt
        options_text = ""
        if options:
            options_text = "\n\nAnswer options:\n" + "\n".join(
                [f"{k}. {v}" for k, v in sorted(options.items())]
            )
        
        system_prompt = """You are an expert at decomposing complex questions into simpler, focused subqueries.
Your task is to generate specific subqueries that, when answered, will help provide a comprehensive answer to the original question.

Guidelines:
1. Each subquery should focus on a specific aspect of the original question
2. Subqueries should be self-contained and searchable
3. Avoid redundancy between subqueries
4. Ensure subqueries cover different perspectives or information needs
5. Make subqueries specific enough to retrieve relevant documents"""
        
        user_prompt = f"""Original Question: {question}{options_text}

Please generate exactly {num_subqueries} focused subqueries that will help answer this question.
Output your response as a JSON array of strings.

Example output format:
["subquery 1", "subquery 2", "subquery 3"]

Generate the subqueries:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            # Try to extract JSON from the response
            content = response.content
            # Find JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                subqueries = json.loads(json_match.group())
            else:
                # Fallback: split by newlines and clean
                subqueries = [
                    line.strip().strip('"').strip("'").strip('-').strip()
                    for line in content.split('\n')
                    if line.strip() and not line.strip().startswith('#')
                ][:num_subqueries]
        except json.JSONDecodeError:
            # Fallback: use the original question as a single subquery
            subqueries = [question]
        
        print(f"\n📝 Generated {len(subqueries)} subqueries:")
        for i, sq in enumerate(subqueries, 1):
            print(f"   {i}. {sq}")
        
        return {"subqueries": subqueries}
    
    def _retrieve_documents(self, query: str, k: int) -> tuple:
        """Retrieve documents for a single query"""
        if self.retrieval_system is None:
            # Mock retrieval for testing
            return [
                {"title": f"Mock Document {i}", "content": f"Mock content for query: {query}"}
                for i in range(k)
            ], [1.0 - i * 0.1 for i in range(k)]
        
        try:
            documents, scores = self.retrieval_system.retrieve(query, k=k)
            return documents, scores
        except Exception as e:
            print(f"Warning: Retrieval failed for '{query}': {e}")
            return [], []
    
    def _generate_subquery_response(
        self,
        subquery: str,
        documents: List[Dict[str, Any]],
        original_question: str,
        options: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a response for a single subquery based on retrieved documents"""
        
        # Build context from documents
        context_parts = []
        for idx, doc in enumerate(documents):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            context_parts.append(f"[Document {idx + 1}] (Title: {title})\n{content}")
        
        context = "\n\n".join(context_parts)
        
        if not context:
            context = "No relevant documents found."
        
        options_text = ""
        if options:
            options_text = "\n\nAnswer options for the original question:\n" + "\n".join(
                [f"{k}. {v}" for k, v in sorted(options.items())]
            )
        
        system_prompt = """You are a helpful assistant that answers questions based on provided documents.
Your task is to answer the subquery using ONLY the information from the provided documents.
Be concise but comprehensive. If the documents don't contain relevant information, say so."""
        
        user_prompt = f"""Original Question: {original_question}{options_text}

Current Subquery: {subquery}

Relevant Documents:
{context}

Based on the documents above, provide a focused answer to the subquery.
Keep in mind that this answer will be used as part of answering the original question.

Answer:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _process_subqueries(self, state: GraphState) -> Dict[str, Any]:
        """Process all subqueries: retrieve documents and generate responses"""
        subqueries = state["subqueries"]
        original_question = state["original_question"]
        options = state.get("options", None)
        k = state.get("k_per_subquery", self.k_per_subquery)
        
        results = []
        
        print(f"\n🔍 Processing {len(subqueries)} subqueries...")
        
        for idx, subquery in enumerate(subqueries, 1):
            print(f"\n   [{idx}/{len(subqueries)}] Processing: {subquery[:50]}...")
            
            # Retrieve documents
            documents, scores = self._retrieve_documents(subquery, k)
            print(f"      Retrieved {len(documents)} documents")
            
            # Generate response
            response = self._generate_subquery_response(
                subquery, documents, original_question, options
            )
            print(f"      Generated response ({len(response)} chars)")
            
            result: SubqueryResult = {
                "subquery": subquery,
                "documents": documents,
                "response": response,
                "scores": scores
            }
            results.append(result)
        
        return {"subquery_results": results}
    
    def _synthesize_answer(self, state: GraphState) -> Dict[str, Any]:
        """Synthesize the final answer from all subquery responses"""
        original_question = state["original_question"]
        options = state.get("options", None)
        subquery_results = state["subquery_results"]
        
        print(f"\n🔄 Synthesizing final answer from {len(subquery_results)} subquery responses...")
        
        # Build the synthesis prompt
        subquery_info = []
        for idx, result in enumerate(subquery_results, 1):
            subquery_info.append(
                f"### Subquery {idx}: {result['subquery']}\n"
                f"**Answer:** {result['response']}\n"
            )
        
        subquery_section = "\n".join(subquery_info)
        
        options_text = ""
        json_format_hint = ""
        if options:
            options_text = "\n\nAnswer options:\n" + "\n".join(
                [f"{k}. {v}" for k, v in sorted(options.items())]
            )
            json_format_hint = '\n\nProvide your answer in JSON format: {"step_by_step_thinking": "your reasoning", "answer_choice": "A/B/C/D"}'
        
        system_prompt = """You are a helpful expert assistant that synthesizes information from multiple sources.
Your task is to provide a comprehensive, well-reasoned answer based on the collected information from subquery responses.

Guidelines:
1. Consider all subquery responses when forming your answer
2. Identify key insights and synthesize them coherently
3. If there are conflicting information, acknowledge and resolve them
4. Provide a clear, definitive answer when possible
5. For multiple-choice questions, clearly indicate your choice"""
        
        user_prompt = f"""Original Question: {original_question}{options_text}

Below are the answers to various subqueries that were generated to help answer the original question:

{subquery_section}

Based on all the information gathered above, please provide a comprehensive answer to the original question.
Think step-by-step and provide your final answer.{json_format_hint}

Final Answer:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        print(f"✅ Final answer generated ({len(response.content)} chars)")
        
        return {"final_answer": response.content}
    
    def answer(
        self,
        question: str,
        options: Optional[Dict[str, str]] = None,
        num_subqueries: Optional[int] = None,
        k_per_subquery: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using the multi-subquery RAG approach.
        
        Args:
            question: The question to answer
            options: Optional dict of answer options (e.g., {"A": "...", "B": "..."})
            num_subqueries: Override default number of subqueries
            k_per_subquery: Override default number of documents per subquery
        
        Returns:
            Dict containing:
                - final_answer: The synthesized answer
                - subqueries: List of generated subqueries
                - subquery_results: Detailed results for each subquery
        """
        print("\n" + "=" * 70)
        print("🚀 Multi-Subquery RAG System")
        print("=" * 70)
        print(f"Question: {question}")
        if options:
            print(f"Options: {options}")
        
        initial_state: GraphState = {
            "original_question": question,
            "options": options,
            "subqueries": [],
            "subquery_results": [],
            "final_answer": "",
            "num_subqueries": num_subqueries or self.num_subqueries,
            "k_per_subquery": k_per_subquery or self.k_per_subquery,
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        print("\n" + "=" * 70)
        print("📊 Results Summary")
        print("=" * 70)
        print(f"Subqueries generated: {len(result['subqueries'])}")
        print(f"Total documents retrieved: {sum(len(r['documents']) for r in result['subquery_results'])}")
        print(f"\n📝 Final Answer:\n{result['final_answer']}")
        print("=" * 70)
        
        return {
            "final_answer": result["final_answer"],
            "subqueries": result["subqueries"],
            "subquery_results": result["subquery_results"],
        }


def main():
    """Example usage of the Multi-Subquery RAG system"""
    
    # Example medical question
    question = "What are the main causes of type 2 diabetes and how can it be prevented?"
    options = {
        "A": "Genetic factors only, no prevention possible",
        "B": "Lifestyle factors, preventable through diet and exercise",
        "C": "Viral infection, preventable through vaccination",
        "D": "Environmental toxins, preventable through avoiding pollutants"
    }
    
    # Initialize the system
    rag_system = MultiSubqueryRAG(
        llm_provider="openai",
        model_name="gpt-4o-mini",
        num_subqueries=3,
        k_per_subquery=5,
    )
    
    # Get the answer
    result = rag_system.answer(question, options)
    
    # Print detailed results
    print("\n\n📋 Detailed Subquery Results:")
    for i, sr in enumerate(result["subquery_results"], 1):
        print(f"\n--- Subquery {i} ---")
        print(f"Query: {sr['subquery']}")
        print(f"Documents: {len(sr['documents'])}")
        print(f"Response: {sr['response'][:200]}...")


if __name__ == "__main__":
    main()

