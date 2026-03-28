# 📚 Multi-Document RAG Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** system that enables intelligent question-answering over multiple PDF documents. Built with LangChain, ChromaDB, Groq LLM, and Streamlit, with built-in RAGAS evaluation.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.2.12-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 📤 **Multi-PDF Upload** - Process multiple documents simultaneously
- 🔍 **Intelligent Retrieval** - Semantic search using HuggingFace embeddings
- 💬 **Context-Aware Answers** - LLM generates answers grounded in your documents
- 📝 **Source Citations** - Every answer includes source file and page number
- 🎨 **Streamlit UI** - Beautiful, user-friendly web interface
- ⚡ **Persistent Vectorstore** - Avoid re-embedding on restart
- 📊 **RAGAS Evaluation** - Built-in automatic quality assessment
- 🚀 **Production Ready** - Full error handling, type hints, and documentation

---

## 🚀 Quick Start

### Prerequisites
- Python 3.14+
- Groq API key (free at https://console.groq.com)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Aryan-3104/MULTI-DOC-RAG.git
cd multi-doc-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file in project root
echo "GROQ_API_KEY=your_api_key_here" > .env
```

### Run the Application

#### **Option 1: Streamlit UI** (Recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

#### **Option 2: Evaluate RAG System**
```bash
python evaluate_rag.py
```
Automatically:
- Loads PDFs from `data/` folder
- Generates test questions
- Runs RAG pipeline
- Evaluates with RAGAS
- Saves results to `evaluation_results.json`

#### **Option 3: Test Evaluation Module**
```bash
python test_evaluator.py
```
Verifies all components are correctly installed.

---

## 📖 Usage

### Via Streamlit Web UI

1. **Upload your PDFs**
   - Drag and drop PDFs in the sidebar
   - Click "Process Documents"
   - Wait for vectorstore creation

2. **Ask Questions**
   - Type your question in the text input
   - Click "Ask Question"
   - View answer with source citations

3. **Example Questions**
   - "What are the main topics covered?"
   - "Explain the key concepts"
   - "How does this compare to other approaches?"

### Via Python Script

```python
from src.loader import load_and_chunk_pdfs
from src.embedder import create_or_load_vectorstore, load_embeddings
from src.retriever import create_retriever
from src.chain import build_rag_chain

# Load documents
chunks = load_and_chunk_pdfs("data")

# Setup vectorstore
embeddings = load_embeddings()
vectorstore = create_or_load_vectorstore(chunks, embeddings)

# Create RAG chain
retriever = create_retriever(vectorstore)
chain = build_rag_chain(retriever)

# Ask a question
response = chain.invoke({"question": "What is the main topic?"})
print(response.content)
```

### Via RAGAS Evaluation

```python
from src.evaluator import run_ragas_evaluation, print_evaluation_report

# Run evaluation with your data
results = run_ragas_evaluation(
    questions=["Question 1?", "Question 2?"],
    answers=["Answer 1", "Answer 2"],
    ground_truths=["Ground truth 1", "Ground truth 2"],
    contexts=[["Context chunk 1"], ["Context chunk 2"]]
)

# Display report
print_evaluation_report(results)
```

---

## 🏗️ Project Structure

```
multi-doc-rag/
├── app.py                          # Streamlit UI application
├── evaluate_rag.py                 # Evaluation orchestrator script
├── test_evaluator.py               # Component verification
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env                           # API keys (NOT in git)
├── .gitignore                     # Git ignore patterns
│
├── src/                           # Core modules
│   ├── __init__.py
│   ├── loader.py                  # PDF loading & chunking
│   ├── embedder.py                # Embeddings & ChromaDB
│   ├── retriever.py               # Document retrieval
│   ├── chain.py                   # LLM chain & prompting
│   └── evaluator.py               # RAGAS evaluation engine
│
├── data/                          # Input PDFs (user uploads)
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
│
└── vectorstore/                   # Persistent vector database
    └── chroma_db/                 # ChromaDB storage
        ├── chroma.sqlite3
        └── [embeddings]/
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required: Groq API key (free tier available)
GROQ_API_KEY=your_groq_api_key_here
```

### Customize Parameters

Edit these files to adjust behavior:

**PDF Chunking** (`src/loader.py`):
```python
chunk_size=500        # Characters per chunk
chunk_overlap=50      # Overlap between chunks
```

**Retrieval** (`src/retriever.py`):
```python
k = max(3, min(num_sources * 4, 15))  # Number of chunks to retrieve
```

**LLM** (`src/chain.py`):
```python
MODEL_NAME = "llama-3.1-8b-instant"   # Groq model
TEMPERATURE = 0                        # 0=deterministic, 1.0=creative
MAX_TOKENS = 1024                      # Response length
```

**Evaluation** (`src/evaluator.py`):
```python
EVALUATOR_MODEL = "mixtral-8x7b-32768"  # Evaluation LLM
NUM_QUESTIONS = 7                        # Test questions in evaluate_rag.py
```

---

## 📊 Core Components

### 1. **Loader** (`src/loader.py`)
Loads PDF files and splits them into chunks.
- **Input**: PDF files from `data/` folder
- **Output**: List of chunked documents with metadata
- **Key Function**: `load_and_chunk_pdfs()`

### 2. **Embedder** (`src/embedder.py`)
Converts text to numerical vectors using HuggingFace embeddings.
- **Model**: `all-MiniLM-L6-v2` (384-dim vectors)
- **Storage**: ChromaDB with SQLite persistence
- **Key Function**: `create_or_load_vectorstore()`

### 3. **Retriever** (`src/retriever.py`)
Finds relevant chunks for a given question via semantic search.
- **Method**: Cosine similarity on embeddings
- **Dynamic k**: Adjusts number of results based on document count
- **Key Function**: `create_retriever()`

### 4. **Chain** (`src/chain.py`)
Orchestrates the RAG pipeline using LangChain LCEL.
- **LLM**: Groq API (llama-3.1-8b-instant)
- **Prompt**: Custom template with source citation instructions
- **Key Function**: `build_rag_chain()`

### 5. **Evaluator** (`src/evaluator.py`)
Measures RAG quality using RAGAS framework.
- **Metrics**: Faithfulness, Answer Relevancy, Context Precision
- **Evaluator LLM**: Groq mixtral-8x7b-32768
- **Key Function**: `run_ragas_evaluation()`

---

## 📈 RAGAS Evaluation

The system includes automatic evaluation using RAGAS (Retrieval-Augmented Generation Assessment).

### Metrics

| Metric | Meaning | Good Score |
|--------|---------|-----------|
| **Faithfulness** | Answer grounded in context | > 0.80 |
| **Answer Relevancy** | Answer addresses question | > 0.85 |
| **Context Precision** | Retrieved docs are relevant | > 0.75 |
| **Overall Score** | Average of all metrics | > 0.80 |

### Run Evaluation

```bash
python evaluate_rag.py
```

### Example Output

```
============================================================
📋 RAGAS EVALUATION REPORT
============================================================

  🎯 Faithfulness:        0.8500 ✓
     └─ Answer grounded in context

  💡 Answer Relevancy:    0.9200 ✓
     └─ Answer relevant to question

  📍 Context Precision:   0.8800 ✓
     └─ Context relevance to question

────────────────────────────────
🏆 Overall Score:        0.8800
   Excellent ⭐⭐⭐⭐⭐
   [████████████████████] 88.0%
============================================================
```

Results are saved to `evaluation_results.json`.

---

## 🔌 Architecture

```
PDFs in data/
    ↓
Loader (chunk documents)
    ↓
Embedder (convert to vectors)
    ↓
ChromaDB (persistent storage)
    ↓
Retriever (semantic search)
    ↓
Chain (format + prompt)
    ↓
Groq LLM (generate answer)
    ↓
Display (with citations)
    ↓
Evaluator (quality assessment)
```

### Data Flow for a Question

```
"What is the main topic?"
    ↓
Convert to embedding (HuggingFace)
    ↓
Search vectorstore (ChromaDB)
    ↓
Get top-k similar chunks
    ↓
Format context with citations
    ↓
Create prompt with context
    ↓
Send to Groq LLM
    ↓
Get streamed response
    ↓
Display with sources
```

---

## 🛠️ Troubleshooting

### "GROQ_API_KEY not found"
**Solution**: Create `.env` file with your Groq API key:
```bash
echo "GROQ_API_KEY=your_key" > .env
```

### "No PDF files found in data/"
**Solution**: Add PDF files to the `data/` folder:
```bash
mkdir data
# Copy your PDFs to this folder
```

### "Vectorstore not found"
**Solution**: Re-process documents:
1. Upload PDFs via Streamlit UI
2. Or run: `python -c "from src.loader import *; from src.embedder import *; chunks = load_and_chunk_pdfs(); embeddings = load_embeddings(); create_vectorstore(chunks, embeddings)"`

### Long first run
**Solution**: First execution caches embeddings (~100MB). Subsequent runs are faster.

### Low evaluation scores
**Solution**: Improve your RAG system:
- Better PDF chunking (adjust `chunk_size`)
- Improve prompts (edit `src/chain.py`)
- Use more/better documents
- Re-run evaluation to measure improvements

### Python 3.14 Pydantic warnings
**Solution**: These warnings are normal and don't affect functionality. They occur due to LangChain's Pydantic V1 compatibility.

---

## 📦 Dependencies

### Core
- **langchain** - LLM orchestration framework
- **langchain-groq** - Groq API integration
- **langchain-chroma** - ChromaDB integration
- **langchain-huggingface** - HuggingFace embeddings

### Vector Database
- **chromadb** - Vector storage and search
- **sentence-transformers** - Embedding model

### LLM Integration
- **groq** - Groq API client
- **langchain_openai** - OpenAI integration (for RAGAS)

### UI
- **streamlit** - Web interface

### Evaluation
- **ragas** - RAG evaluation framework
- **datasets** - Data handling

### Utilities
- **python-dotenv** - Environment variables
- **PyPDF2** - PDF processing

See `requirements.txt` for full list with versions.

---

## 🚀 Performance Tips

1. **Faster Responses**
   - Use smaller `chunk_size` for more precise retrieval
   - Reduce `MAX_TOKENS` if not needed

2. **Better Quality**
   - Increase `chunk_overlap` for context continuity
   - Use larger evaluation LLM (mixtral vs llama)

3. **Cost Optimization**
   - Groq free tier includes all features
   - Reuse vectorstore (no re-embedding)

4. **Resource Management**
   - HuggingFace embeddings run locally (CPU)
   - ChromaDB persists to avoid recomputation
   - Streamlit session caching for speed

---

## 🧪 Testing

### Run Verification Tests
```bash
python test_evaluator.py
```

### Test Individual Modules
```python
# Test loader
from src.loader import load_and_chunk_pdfs
chunks = load_and_chunk_pdfs()
print(f"Loaded {len(chunks)} chunks")

# Test embedder
from src.embedder import load_embeddings
embeddings = load_embeddings()
print(f"Embeddings loaded: {embeddings.model_name}")

# Test evaluator
from src.evaluator import create_sample_test_data, run_ragas_evaluation
sample = create_sample_test_data()
results = run_ragas_evaluation(**sample)
print(results)
```

---

## 📚 Documentation

- **README.md** - This file (project overview)
- **EVALUATION.md** - Detailed evaluation guide
- **QUICK_REFERENCE.md** - Quick lookup for common tasks
- **Inline Docstrings** - Full documentation in source code

---

## 🔐 Security

- `.env` file is in `.gitignore` (API keys not pushed)
- No sensitive data in code
- PDFs in `data/` are local-only
- Vectorstore on local machine

### Best Practices
- Keep `.env` file secure
- Don't commit `.env` to git
- Rotate API keys periodically
- Use `.env.example` as template

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🎯 Roadmap

- [ ] Multi-language support
- [ ] Database integration (PostgreSQL with pgvector)
- [ ] Advanced retrieval (re-ranking, filtering)
- [ ] Conversation history
- [ ] Multiple LLM support
- [ ] Query optimization
- [ ] Web deployment (Docker)

---

## 💡 Examples

### Example 1: Basic Q&A
```bash
# 1. Upload PDFs via Streamlit UI
# 2. Ask: "What are the main topics?"
# 3. Get cited answer with sources
```

### Example 2: Batch Evaluation
```bash
# Evaluate on 50 questions
python evaluate_rag.py

# Check results
cat evaluation_results.json
```

### Example 3: Custom Integration
```python
from src.evaluator import run_ragas_evaluation, print_evaluation_report

# Your data
qs = ["Q1?", "Q2?"]
as_ = ["A1", "A2"]
gts = ["GT1", "GT2"]
ctx = [["C1"], ["C2"]]

# Evaluate
results = run_ragas_evaluation(qs, as_, gts, ctx)
print_evaluation_report(results)
```

---

## 📞 Support

- Check `QUICK_REFERENCE.md` for common issues
- Read `EVALUATION.md` for evaluation specifics
- See docstrings in `src/` for API details
- Run `test_evaluator.py` to verify setup

---

## 🎉 Getting Started

```bash
# 1. Clone and setup
git clone <repo>
cd multi-doc-rag
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Add API key
echo "GROQ_API_KEY=your_key" > .env

# 3. Add PDFs
mkdir data
# Copy your PDFs to data/

# 4. Run
streamlit run app.py

# 5. Evaluate (optional)
python evaluate_rag.py
```

That's it! You now have a production-ready RAG system. 🚀

---

**Created**: March 2026  
**Python Version**: 3.14  
**Framework**: LangChain + RAGAS + Groq  
**Status**: ✅ Production Ready

For detailed information, see the documentation files or check the source code docstrings.
