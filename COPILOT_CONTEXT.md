# Project Context for GitHub Copilot

## Project Name
Multi-Document Research Assistant (RAG-based)

## Goal
Build a RAG (Retrieval-Augmented Generation) application that allows users to 
upload multiple PDFs and ask questions across them with source attribution.

## Tech Stack
- Framework: LangChain
- Vector Database: ChromaDB (local, persistent)
- Embedding Model: all-MiniLM-L6-v2 (HuggingFace, runs locally, no API cost)
- LLM: Llama 3.1 8B via Groq API (free tier)
- UI: Streamlit
- Evaluation: RAGAS
- Deployment: Streamlit Cloud

## Project Structure
multi-doc-rag/
├── data/                      # PDFs go here
├── vectorstore/               # ChromaDB persists here
├── src/
│   ├── loader.py              # Load + chunk PDFs
│   ├── embedder.py            # Embed + store in ChromaDB
│   ├── retriever.py           # Query vectorstore
│   └── chain.py               # LLM + prompt logic
├── app.py                     # Streamlit UI
├── requirements.txt
└── .env                       # Groq API key

## Key Decisions & Reasons
- ChromaDB: local, no setup, persistent, metadata filtering built-in
- all-MiniLM-L6-v2: lightweight, CPU-friendly, free, no rate limits
- Groq: fastest free LLM API, no credit card needed
- Streamlit: pure Python UI, fast to build, easy to deploy free
- RAGAS: industry-standard RAG evaluation framework

## Chunking Strategy
- Splitter: RecursiveCharacterTextSplitter
- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Metadata per chunk: source filename + page number

## RAG Pipeline Flow
User Question
    → Convert to embedding vector
    → Search ChromaDB for top 4 similar chunks
    → Format chunks with source metadata
    → Pass to Llama 3.1 via Groq with prompt
    → Return answer with source citations

## Prompt Rules
- Always cite which document the answer came from
- Synthesize across multiple documents when needed
- Say "I could not find this" if context is insufficient
- Temperature = 0 for factual, deterministic answers

## Evaluation Metrics (RAGAS)
- Faithfulness: is the answer grounded in retrieved context?
- Answer Relevance: does the answer address the question?
- Context Precision: were the right chunks retrieved?

## Constraints
- 100% free stack, no paid APIs
- Beginner-friendly, no Docker, no cloud databases
- Must run locally and deploy on Streamlit Cloud
```

---

