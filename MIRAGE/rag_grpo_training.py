#!/usr/bin/env python3
"""
RAG Scenario GRPO Training for MedGemma
Uses MedRAG with MedQA dataset and trains with GRPO using correctness as reward
"""
import unsloth
import os
import sys
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
from datasets import Dataset
import argparse

# Add paths for MedRAG and src
sys.path.insert(0, '.')
sys.path.insert(0, 'MedRAG')
sys.path.insert(0, 'MedRAG/src')
sys.path.insert(0, 'src')

# Import MedRAG
try:
    from MedRAG.src.medrag import MedRAG
    from MedRAG.src.template import general_medrag_system, general_medrag
    print("✓ Successfully imported MedRAG")
except ImportError as e:
    print(f"✗ Failed to import MedRAG: {e}")
    sys.exit(1)

# Import QADataset
try:
    # Remove MedRAG/src from path temporarily to avoid conflict
    medrag_src_path = 'MedRAG/src'
    if medrag_src_path in sys.path:
        sys.path.remove(medrag_src_path)
    
    from src.utils import QADataset
    print("✓ Successfully imported QADataset")
    
    # Add MedRAG/src back
    sys.path.append(medrag_src_path)
    
except ImportError as e:
    print(f"✗ Failed to import QADataset: {e}")
    # Try alternative approach
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("qa_utils", "src/utils.py")
        qa_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qa_utils)
        QADataset = qa_utils.QADataset
        print("✓ Successfully imported QADataset using importlib")
    except Exception as e2:
        print(f"✗ Alternative import also failed: {e2}")
        sys.exit(1)

# Import unsloth and TRL
try:
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    print("✓ Successfully imported unsloth and TRL")
except ImportError as e:
    print(f"✗ Failed to import unsloth/TRL: {e}")
    print("Please install: pip install unsloth[colab-new] @ git+https://github.com/huggingface/trl.git")
    sys.exit(1)

def parse_answer(answer_text):
    """
    Parse answer from model output to extract the choice (A, B, C, D)
    """
    try:
        # Try to parse as JSON first
        if isinstance(answer_text, str):
            # Look for JSON pattern
            if '{' in answer_text and '}' in answer_text:
                start = answer_text.find('{')
                end = answer_text.rfind('}') + 1
                json_str = answer_text[start:end]
                answer_data = json.loads(json_str)
                return answer_data.get('answer_choice', answer_data.get('answer', ''))
        else:
            answer_data = answer_text
            return answer_data.get('answer_choice', answer_data.get('answer', ''))
    except:
        pass
    
    # Fallback: look for patterns like "answer is A" or just "A"
    answer_text = str(answer_text).upper()
    for choice in ['A', 'B', 'C', 'D']:
        if f"ANSWER IS {choice}" in answer_text or f"CHOICE {choice}" in answer_text:
            return choice
        if f"ANSWER: {choice}" in answer_text or f"ANSWER_CHOICE\": \"{choice}" in answer_text:
            return choice
    
    # Last resort: find isolated A, B, C, D
    matches = re.findall(r'\b([ABCD])\b', answer_text)
    if matches:
        return matches[-1]  # Take the last match
    
    return ""

def prepare_medqa_dataset_with_rag(max_samples=None, corpus_name="Textbooks"):
    """
    Prepare MedQA dataset for GRPO training with actual RAG retrieval
    """
    print("Loading MedQA dataset...")
    try:
        medqa_dataset = QADataset("medqa")
        total_questions = len(medqa_dataset)
        print(f"✓ MedQA dataset loaded: {total_questions} questions")
    except Exception as e:
        print(f"✗ Error loading MedQA dataset: {e}")
        return None
    
    # Limit samples if specified
    if max_samples:
        total_questions = min(max_samples, total_questions)
        print(f"Using first {total_questions} questions for training")
    
    # Initialize MedRAG for document retrieval only (no LLM needed)
    print(f"\nInitializing MedRAG retrieval system with {corpus_name} corpus...")
    try:
        # Import RetrievalSystem directly to avoid loading LLM
        from MedRAG.src.utils import RetrievalSystem
        
        retrieval_system = RetrievalSystem(
            retriever_name="MedCPT",
            corpus_name=corpus_name,
            db_dir="./MedRAG/corpus",
            cache=True
        )
        print(f"✓ MedRAG retrieval system with {corpus_name} initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize MedRAG retrieval system: {e}")
        return None
    
    # Convert to format expected by GRPO with actual RAG retrieval
    data = []
    print(f"\nRetrieving documents for {total_questions} questions...")
    
    for i in tqdm(range(total_questions), desc="Processing questions with RAG"):
        try:
            question_data = medqa_dataset[i]
            
            # Get question and options
            question = question_data['question']
            options = question_data['options']
            correct_answer = question_data['answer']
            
            # Retrieve relevant documents using MedRAG retrieval system
            try:
                # Use retrieval system to get relevant documents (same as MedRAG)
                retrieved_snippets, scores = retrieval_system.retrieve(
                    question=question,
                    k=32,  # Retrieve top 32 documents
                    rrf_k=100  # Same as MedRAG default
                )
                
                # Format retrieved documents as context (same format as MedRAG)
                if retrieved_snippets and len(retrieved_snippets) > 0:
                    contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(
                        idx, 
                        retrieved_snippets[idx]["title"], 
                        retrieved_snippets[idx]["content"]
                    ) for idx in range(len(retrieved_snippets))]
                    
                    # Use all retrieved documents (same as MedRAG)
                    context = "\n\n".join(contexts)
                else:
                    print(f"Warning: No relevant documents found for question {i}")
                    context = "No relevant documents found. Please use your medical knowledge to answer the question."
                    
            except Exception as e:
                print(f"Warning: Failed to retrieve documents for question {i}: {e}")
                context = "Document retrieval failed. Please use your medical knowledge to answer the question."
            
            # Format options as string (same format as in MedRAG)
            options_text = ""
            for key, value in options.items():
                options_text += f"{key}. {value}\n"
            
            # Use the general_medrag template format with actual retrieved context
            formatted_question = general_medrag.render(
                context=context,
                question=question,
                options=options_text.strip()
            )
            
            data.append({
                'prompt': [
                    {'role': 'system', 'content': general_medrag_system},
                    {'role': 'user', 'content': formatted_question}
                ],
                'answer': correct_answer,
                'question_id': i,
                'context_length': len(context)
            })
            
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            continue
    
    print(f"✓ Successfully prepared {len(data)} samples with RAG documents")
    return Dataset.from_list(data)

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward function based on correctness of the answer
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [parse_answer(r) for r in responses]
    
    print('-'*20, f"Question:\n{q[:200]}...", f"\nCorrect Answer:\n{answer[0]}", 
          f"\nResponse:\n{responses[0][:200]}...", f"\nExtracted:\n{extracted_responses[0]}")
    
    # Reward: 1.0 for correct answer, 0.0 for incorrect
    rewards = []
    for extracted, correct in zip(extracted_responses, answer):
        if extracted.strip().upper() == correct.strip().upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    return rewards

def setup_model_and_tokenizer(model_name="google/medgemma-4b-it", max_seq_length=2048):
    """
    Setup MedGemma model with unsloth for GRPO training
    """
    print(f"Setting up model: {model_name}")
    
    lora_rank = 32
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print("✓ Model and tokenizer setup complete")
    return model, tokenizer

def train_rag_grpo(model, tokenizer, dataset, output_dir="rag_grpo_outputs", 
                   num_epochs=1, batch_size=8, learning_rate=5e-6, max_steps=500):
    """
    Train the model using GRPO with RAG scenario
    """
    print("Setting up GRPO training...")
    
    max_prompt_length = 512  # Increased for medical questions
    
    training_args = GRPOConfig(
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        num_generations=6,  # Reduced for memory efficiency
        max_prompt_length=max_prompt_length,
        max_completion_length=2048 - max_prompt_length,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        save_steps=50,
        max_grad_norm=0.1,
        report_to=None,  # Disable tensorboard to avoid dependency issues
        output_dir=output_dir,
        remove_unused_columns=False,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward_func],  # Only correctness reward
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting GRPO training...")
    trainer.train()
    
    print("✓ Training completed!")
    return trainer

def evaluate_model(model, tokenizer, test_dataset, num_samples=100):
    """
    Evaluate the trained model on test data
    """
    print(f"Evaluating model on {num_samples} samples...")
    
    correct_count = 0
    total_count = 0
    
    for i in tqdm(range(min(num_samples, len(test_dataset)))):
        try:
            sample = test_dataset[i]
            prompt = sample['prompt']
            correct_answer = sample['answer']
            
            # Generate response
            text = tokenizer.apply_chat_template(
                prompt, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Removed vLLM dependency for evaluation
            
            # Use standard generate method for evaluation
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            output = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse and check answer
            predicted_answer = parse_answer(output)
            is_correct = predicted_answer.strip().upper() == correct_answer.strip().upper()
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            if (i + 1) % 20 == 0:
                current_accuracy = correct_count / total_count * 100
                print(f"Progress: {i+1}/{num_samples}, Accuracy: {current_accuracy:.2f}%")
                
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            continue
    
    final_accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    print(f"\nFinal Evaluation Results:")
    print(f"Correct: {correct_count}/{total_count}")
    print(f"Accuracy: {final_accuracy:.2f}%")
    
    return final_accuracy

def main():
    parser = argparse.ArgumentParser(description='RAG GRPO Training for MedGemma')
    parser.add_argument('--model', type=str, default="google/medgemma-4b-it", 
                       help='Model name to use')
    parser.add_argument('--max-samples', type=int, default=2000, 
                       help='Maximum number of training samples')
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of training epochs')
    parser.add_argument('--max-steps', type=int, default=500, 
                       help='Maximum number of training steps')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-6, 
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for training (default: auto-generated with timestamp)')
    parser.add_argument('--eval-samples', type=int, default=100, 
                       help='Number of samples for evaluation')
    parser.add_argument('--eval-only', action='store_true', 
                       help='Only run evaluation (skip training)')
    parser.add_argument('--corpus', type=str, default="Textbooks", 
                       choices=["Textbooks", "PubMed"], 
                       help='Corpus to use for RAG retrieval')
    
    args = parser.parse_args()
    
    # Generate output directory with timestamp if not provided
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"rag_grpo_outputs_{timestamp}"
    
    print("=" * 80)
    print("RAG GRPO Training for MedGemma")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Corpus: {args.corpus}")
    print(f"Max samples: {args.max_samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Prepare dataset with RAG
    dataset = prepare_medqa_dataset_with_rag(max_samples=args.max_samples, corpus_name=args.corpus)
    if dataset is None:
        print("Failed to prepare dataset with RAG")
        sys.exit(1)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(args.model)
    
    if not args.eval_only:
        # Train model
        trainer = train_rag_grpo(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps
        )
        
        # Save the trained model using unsloth's save_pretrained method
        model.save_pretrained(f"{args.output_dir}/rag_grpo_lora")
        print(f"✓ Model saved to {args.output_dir}/rag_grpo_lora")
    
    # Evaluate model
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    # Load test dataset for evaluation with RAG
    test_dataset = prepare_medqa_dataset_with_rag(max_samples=args.eval_samples + 100, corpus_name=args.corpus)  # Get more for test
    if test_dataset:
        # Use samples after training data for evaluation
        eval_start = min(args.max_samples, len(test_dataset) - args.eval_samples)
        eval_dataset = test_dataset.select(range(eval_start, eval_start + args.eval_samples))
        
        accuracy = evaluate_model(model, tokenizer, eval_dataset, args.eval_samples)
        
        # Save evaluation results
        results = {
            'model': args.model,
            'training_samples': args.max_samples,
            'evaluation_samples': args.eval_samples,
            'accuracy': accuracy,
            'timestamp': int(time.time())
        }
        
        results_file = f"{args.output_dir}/evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Evaluation results saved to {results_file}")
    
    print("\n✓ RAG GRPO training completed successfully!")

if __name__ == "__main__":
    main()
