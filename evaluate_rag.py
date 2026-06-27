#!/usr/bin/env python3
"""
RAG Evaluation Script

Loads PDFs, generates test questions, runs RAG pipeline, and evaluates with RAGAS.
Saves evaluation results to evaluation_results.json.

Usage:
    python evaluate_rag.py
"""

import nest_asyncio
nest_asyncio.apply = lambda: None

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Reconfigure stdout/stderr to UTF-8 to prevent UnicodeEncodeError on Windows console
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Import RAG components
from src.loader import load_and_chunk_pdfs
from src.embedder import load_embeddings, create_or_load_vectorstore
from src.retriever import create_retriever, format_retrieved_chunks
from src.chain import create_prompt
from src.evaluator import (
    run_ragas_evaluation,
    print_evaluation_report,
    get_evaluator_llm
)

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "data"
RESULTS_FILE = "evaluation_results.json"
NUM_QUESTIONS = 3  # 3 questions keeps usage within Groq free-tier limits


def get_groq_api_key() -> str:
    """
    Get Groq API key from environment.

    Raises:
        ValueError: If GROQ_API_KEY not set
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Please add: GROQ_API_KEY=your_key"
        )
    return api_key


def create_groq_llm():
    """
    Create a ChatGroq LLM instance for question generation.

    Uses llama-3.3-70b-versatile — high quality, generous free tier.

    Returns:
        ChatGroq LLM object
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )


def generate_test_questions(chunks: list) -> list:
    """
    Generate test questions from document chunks using LLM.
    
    Args:
        chunks: List of document chunks
    
    Returns:
        List of generated questions
    """
    print(f"\n🎯 Generating {NUM_QUESTIONS} test questions from documents...")
    
    try:
        # Prepare document summary
        sample_chunks = chunks[:5]  # Use first 5 chunks as context
        doc_summary = "\n\n".join([chunk.page_content for chunk in sample_chunks])
        
        # Get LLM
        llm = create_groq_llm()
        
        # Create prompt for question generation
        question_generation_prompt = f"""Based on the following document excerpts, generate exactly {NUM_QUESTIONS} evaluation questions.

RULES — follow ALL of these strictly:
1. One sentence only. Maximum 15 words.
2. Ask about a SPECIFIC process, mechanism, requirement, property, or count found in the document.
   - NOT a vague definition: do NOT ask "What is X?" or "Define X."
   - NOT a compound question: do NOT use "and", "as well as", or ask two things at once.
3. The question must name a specific concept from the document AND ask something concrete about it (how it works, how many steps/requirements, which type is used, what it contains, etc.).
4. Each question must be fully answerable from the document text alone.
5. Each question must be on a different topic.
6. Output one question per line. No numbering, no bullets, no extra text.

GOOD examples (specific, process-level):
- What are the five requirements of a consensus mechanism?
- Which consensus type is used in permissioned blockchains?
- What does the block header store in a blockchain?
- How does Proof of Work select the next block proposer?

BAD examples (too vague or definitional):
- What is consensus?
- What is a block?
- What is Byzantine Fault Tolerance?
- What is blockchain technology?

Document Excerpts:
{doc_summary}

Generate {NUM_QUESTIONS} specific, process-level questions:"""

        
        # Generate questions
        response = llm.invoke(question_generation_prompt)
        questions_text = response.content
        
        # Parse questions from response
        questions = [
            q.strip() 
            for q in questions_text.split("\n") 
            if q.strip() and len(q.strip()) > 10
        ]
        
        # Ensure we have enough questions
        questions = questions[:NUM_QUESTIONS]
        
        if not questions:
            print("⚠️  Could not generate questions. Using default questions.")
            questions = [
                "What are the main topics covered in the documents?",
                "What is the significance of the information presented?",
                "What are the key concepts discussed?",
                "How is the content organized and structured?",
                "What practical applications are mentioned?",
                "What are the key findings or conclusions?",
                "What technologies or methods are discussed?"
            ][:NUM_QUESTIONS]
        
        print(f"✓ Generated {len(questions)} questions:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
        
        return questions
        
    except Exception as e:
        print(f"⚠️  Error generating questions: {e}")
        print("Using default questions instead.")
        return [
            "What are the main topics covered in the documents?",
            "What is the significance of the information presented?",
            "What are the key concepts discussed?",
            "How is the content organized and structured?",
            "What practical applications are mentioned?",
            "What are the key findings or conclusions?",
            "What technologies or methods are discussed?"
        ][:NUM_QUESTIONS]


def run_rag_pipeline(
    questions: list,
    retriever,
    llm
) -> tuple:
    """
    Run RAG pipeline for each question.
    
    Retrieves context chunks and generates answers.
    Also collects the retrieved contexts for RAGAS evaluation.
    
    Args:
        questions: List of evaluation questions
        retriever: LangChain retriever
        llm: LangChain LLM (should have invoke method)
    
    Returns:
        Tuple of:
        - answers: list of generated answers
        - contexts: list of lists of retrieved context chunks
    """
    print("\n🔄 Running RAG pipeline for each question...")
    
    answers = []
    contexts = []
    prompt_template = create_prompt()
    
    for i, question in enumerate(questions, 1):
        try:
            print(f"\n  [{i}/{len(questions)}] {question[:60]}...")
            
            # Retrieve relevant chunks
            retrieved_docs = retriever.invoke(question)
            
            # Format context
            context_text = format_retrieved_chunks(retrieved_docs)
            
            # Store raw context chunks for RAGAS
            context_chunks = [doc.page_content for doc in retrieved_docs]
            contexts.append(context_chunks)
            
            # Create full prompt
            prompt = prompt_template.format(context=context_text, question=question)
            
            # Generate answer
            response = llm.invoke(prompt)
            answer = response.content
            
            answers.append(answer)
            print(f"     ✓ Generated answer ({len(answer)} chars)")
            
        except Exception as e:
            print(f"     ❌ Error: {e}")
            answers.append("Unable to generate answer.")
            contexts.append([])
    
    print(f"\n✓ RAG pipeline completed for {len(answers)} questions")
    return answers, contexts


def create_ground_truths(questions: list, answers: list) -> list:
    """
    Create ground truth answers using LLM.
    
    Uses LLM to generate reference answers that aren't based on retrieval.
    These serve as the ground truth for evaluation.
    
    Args:
        questions: List of evaluation questions
        answers: List of RAG-generated answers (for reference)
    
    Returns:
        List of ground truth answers
    """
    print("\n📚 Creating ground truth answers...")
    
    try:
        llm = create_groq_llm()
        ground_truths = []
        
        for i, question in enumerate(questions, 1):
            try:
                # Generate ground truth without document context
                # Just based on general knowledge
                prompt = f"""Answer this question based on your knowledge. Provide a clear, concise answer without citations:

Question: {question}

Answer:"""
                
                response = llm.invoke(prompt)
                ground_truth = response.content.strip()
                ground_truths.append(ground_truth)
                print(f"  {i}. ✓ Ground truth created")
                
            except Exception as e:
                print(f"  {i}. ⚠️  Error: {e}")
                ground_truths.append("Unable to generate ground truth.")
        
        return ground_truths
        
    except Exception as e:
        print(f"⚠️  Error creating ground truths: {e}")
        # Return dummy ground truths if generation fails
        return [f"Reference answer for: {q}" for q in questions]


def save_evaluation_results(results: dict, answers: list[str], questions: list[str]) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: RAGAS evaluation results
        answers: List of generated answers
        questions: List of evaluation questions
    """
    import time
    from datetime import datetime, timezone, timedelta

    IST = timezone(timedelta(hours=5, minutes=30))
    now = time.time()
    output = {
        "timestamp": now,
        "datetime": datetime.fromtimestamp(now, tz=IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "metrics": results,
        "num_questions": len(questions),
        "num_answers": len(answers),
        "evaluation_data": {
            "questions": questions,
            "answers": answers[:3] if len(answers) > 3 else answers  # Save first 3 for brevity
        }
    }
    
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """
    Main evaluation workflow.
    """
    print("\n" + "=" * 70)
    print("🚀 RAG EVALUATION PIPELINE".center(70))
    print("=" * 70)
    
    try:
        # 1. Validate API key
        print("\n🔑 Validating API credentials...")
        get_groq_api_key()
        print("✓ Groq API key found")
        
        # 2. Load and chunk PDFs
        print(f"\n📄 Loading PDFs from '{DATA_DIR}'...")
        chunks = load_and_chunk_pdfs(DATA_DIR)
        
        if not chunks:
            print("❌ No documents loaded. Please add PDF files to the data/ folder.")
            sys.exit(1)
        
        print(f"✓ Loaded {len(chunks)} chunks")
        
        # 3. Load embeddings and create/load vectorstore
        print("\n🧠 Setting up embeddings and vectorstore...")
        embeddings = load_embeddings()
        vectorstore = create_or_load_vectorstore(chunks, embeddings)
        print("✓ Vectorstore ready")
        
        # 4. Create retriever
        print("\n🔍 Creating retriever...")
        retriever = create_retriever(vectorstore)
        print("✓ Retriever ready")
        
        # 5. Generate test questions
        questions = generate_test_questions(chunks)
        
        # 6. Run RAG pipeline
        llm = create_groq_llm()
        answers, contexts = run_rag_pipeline(questions, retriever, llm)
        
        # 7. Create ground truths
        ground_truths = create_ground_truths(questions, answers)
        
        # 8. Run RAGAS evaluation
        print("\n" + "=" * 70)
        results = run_ragas_evaluation(questions, answers, ground_truths, contexts)
        
        # 9. Print evaluation report
        print_evaluation_report(results)
        
        # 10. Save results
        save_evaluation_results(results, answers, questions)
        
        print("=" * 70)
        print("✅ Evaluation completed successfully!".center(70))
        print("=" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
