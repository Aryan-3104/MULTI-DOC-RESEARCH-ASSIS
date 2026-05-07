# Multi-Document RAG

Multi-Document RAG is a Streamlit app for uploading PDF files, searching them with embeddings, and asking questions with answers grounded in the retrieved context. It also includes a separate evaluation pipeline that scores the RAG system with RAGAS.

## Clone the repo

Use your GitHub username when cloning:

```powershell
git clone https://github.com/Aryan-3104/multi-doc-rag.git
cd multi-doc-rag
```

## Setup

This project is meant to run on Windows with Python and a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq key:

```text
GROQ_API_KEY=your_groq_api_key_here
```

## Run the app

Start the Streamlit UI:

```powershell
streamlit run app.py
```

Then:
1. Upload one or more PDF files.
2. Click **Process Documents**.
3. Ask questions in the main panel.

## How evaluation works

Run the evaluation script when you want to measure RAG quality:

```powershell
python evaluate_rag.py
```

The evaluation pipeline does this:
1. Loads PDFs from `data/`.
2. Splits them into chunks and builds the vector store.
3. Generates test questions from the document content.
4. Runs the RAG pipeline to produce answers and retrieve context.
5. Creates reference answers.
6. Scores the results with RAGAS metrics: faithfulness, answer relevancy, and context precision.
7. Saves the results to `evaluation_results.json`.

If RAGAS is unavailable, the code falls back to a simple heuristic scoring method.

## Important files

- `app.py` - Streamlit UI for upload and Q&A
- `evaluate_rag.py` - Batch evaluation runner
- `src/loader.py` - PDF loading and chunking
- `src/embedder.py` - Embeddings and vector store setup
- `src/retriever.py` - Similarity search over chunks
- `src/chain.py` - RAG prompt and LLM chain
- `src/evaluator.py` - RAGAS scoring and fallback logic
- `test_evaluator.py` - Evaluator tests

## Data and storage

- `data/` stores uploaded PDFs locally.
- `vectorstore/` stores ChromaDB files locally.
- Both should stay out of git.

## Troubleshooting

If the app does not start or evaluation fails:

1. Check that `GROQ_API_KEY` is set correctly in `.env`.
2. Make sure PDFs exist in `data/` before running evaluation.
3. Delete `vectorstore/chroma_db/` if you want to force a rebuild.
4. Reinstall dependencies if imports fail:

```powershell
pip install -r requirements.txt
```

## Notes

This repo uses Groq for the LLM, ChromaDB for storage, LangChain for orchestration, Streamlit for the UI, and RAGAS for evaluation.
