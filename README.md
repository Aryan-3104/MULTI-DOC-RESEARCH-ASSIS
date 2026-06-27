# Multi-Document RAG

Multi-Document RAG is a Streamlit app for uploading PDF files, searching them with embeddings, and asking questions with answers grounded in the retrieved context. It includes a complete evaluation pipeline that scores the RAG system using real RAGAS metrics.

## Clone the repo

```powershell
git clone https://github.com/Aryan-3104/multi-doc-rag.git
cd multi-doc-rag
```

## Setup

This project runs on Windows with Python and a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
# Main app (chain.py) — Gemini 2.5 Flash
GOOGLE_API_KEY=your_google_api_key_here

# Evaluation pipeline (evaluate_rag.py + src/evaluator.py) — Groq
GROQ_API_KEY=your_groq_api_key_here
```

## Run the app

```powershell
streamlit run app.py
```

Then:
1. Upload one or more PDF files.
2. Click **Process Documents**.
3. Ask questions in the main panel.

## How evaluation works

Place your PDFs inside `data/` and run:

```powershell
python evaluate_rag.py
```

The pipeline:
1. Loads PDFs from `data/` and chunks them (800 chars, 150 overlap).
2. Builds or reloads the ChromaDB vectorstore.
3. Generates 3 test questions from document content.
4. Runs the RAG pipeline: MMR retrieval (k=3) → strict context-only prompt → LLM answer.
5. Creates reference ground-truth answers.
6. Scores with real RAGAS metrics: faithfulness, answer relevancy, context precision.
7. Saves results to `evaluation_results.json`.

> Results are extracted via pandas DataFrame for compatibility with RAGAS 0.2+.

### Latest benchmark (BlockChain PDF, 229 chunks)

| Metric | Score |
|---|---|
| 🎯 Faithfulness | 0.9167 |
| 💡 Answer Relevancy | 0.9677 |
| 📍 Context Precision | 1.0000 |
| 🏆 **Overall** | **0.9615 — Excellent ⭐⭐⭐⭐⭐** |

## Important files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI for upload and Q&A |
| `evaluate_rag.py` | Batch evaluation orchestrator |
| `src/loader.py` | PDF loading and chunking (800 char chunks) |
| `src/embedder.py` | all-MiniLM-L6-v2 embeddings + ChromaDB |
| `src/retriever.py` | MMR retrieval with dynamic k |
| `src/chain.py` | Strict context-only RAG prompt + Gemini 2.5 Flash LLM |
| `src/evaluator.py` | RAGAS scoring with pandas-based extraction |
| `test_evaluator.py` | Component verification script |

## Data and storage

- `data/` — PDF files for evaluation and Q&A.
- `vectorstore/` — ChromaDB persisted on disk.
- Both are excluded from git via `.gitignore`.

> **After changing chunking settings**, delete `vectorstore/chroma_db/` to force a rebuild:
> ```powershell
> Remove-Item -Recurse -Force vectorstore\chroma_db
> ```

## Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY not found` | Add key to `.env` file (needed by `app.py`) |
| `GROQ_API_KEY not found` | Add key to `.env` file (needed by `evaluate_rag.py`) |
| `No PDF files found` | Add PDFs to `data/` folder |
| `rate_limit_exceeded` | Reduce `NUM_QUESTIONS` in `evaluate_rag.py` or wait for daily limit reset |
| `create_prompt() got an unexpected keyword argument 'mode'` | ✅ Fixed — removed stale `mode="eval"` arg from `evaluate_rag.py` line 169 |
| Low Context Precision | Lower `k` in `retriever.py` or switch to MMR (already default) |
| Pydantic V1 warnings | Expected on Python 3.14 — safe to ignore |
| Stale vectorstore after config change | Delete `vectorstore/chroma_db/` and re-run |

## Stack

- **App LLM**: Google Gemini (`gemini-2.5-flash`) — via `chain.py`
- **Eval LLM**: Groq (`llama-3.3-70b-versatile`) — via `evaluate_rag.py` + `src/evaluator.py`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key needed)
- **Vector DB**: ChromaDB (persisted locally)
- **Retrieval**: MMR (Maximal Marginal Relevance)
- **Evaluation**: RAGAS 0.2+ with pandas result extraction
- **UI**: Streamlit
- **Orchestration**: LangChain
