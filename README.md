# Multi-Document RAG 📚🤖

A production-ready **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and ask intelligent questions about them. Built with LangChain, ChromaDB, and Groq LLM on a 100% free stack.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app)

---

## ✨ Features

- 📄 **Multi-document upload** - Process multiple PDFs simultaneously
- 🔍 **Semantic search** - Find relevant content using AI embeddings (all-MiniLM-L6-v2)
- 💬 **Intelligent Q&A** - Ask questions and get context-aware answers
- ⚡ **Fast responses** - Powered by Groq's free LLM API (8B and 32B models)
- 🎨 **Clean UI** - Professional Streamlit interface
- 📊 **Quality evaluation** - RAGAS metrics for answer quality (faithfulness, relevancy, precision)
- 🚀 **Cloud ready** - One-click deployment on Streamlit Cloud
- 💾 **Persistent storage** - ChromaDB for efficient vector storage and retrieval

---

## 🛠 Tech Stack

| Component | Technology | Why This Choice |
|-----------|-----------|-----------------|
| **UI Framework** | Streamlit 1.55.0 | Fast to build, reactive components, perfect for demos |
| **LLM Framework** | LangChain 1.2.12 | Industry standard, LCEL chains, excellent integrations |
| **Vector Database** | ChromaDB 1.5.5 | Lightweight, no maintenance, SQLite-based persistence |
| **Embeddings** | HuggingFace all-MiniLM-L6-v2 | 384-dim, fast, cached locally, 100% free |
| **Language Model** | Groq llama-3.1-8b-instant | Free tier, fast inference, production-ready |
| **PDF Processing** | PyPDF 6.9.1 | Pure Python, no dependencies, reliable |
| **Evaluation** | RAGAS 0.3.0 | Standard RAG quality metrics |
| **Cost** | 💰 **$0/month** | Groq free tier + Streamlit Community Cloud free |

---

## 🚀 Quick Start

### Local Development (Windows)

#### 1. Clone the Repository
```powershell
git clone https://github.com/YOUR-USERNAME/multi-doc-rag.git
cd multi-doc-rag
```

#### 2. Create Virtual Environment
```powershell
# Create environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1
```

#### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

#### 4. Get a Free Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free)
3. Copy your API key

#### 5. Create `.env` File
```powershell
# Create .env in the project root
"GROQ_API_KEY=your_api_key_here" | Out-File .env -Encoding UTF8
```

#### 6. Run Locally
```powershell
streamlit run app.py
```
Your app opens at `http://localhost:8501` 🎉

---

## ☁️ Deploy on Streamlit Cloud (Free)

### 1. Push to GitHub
```powershell
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push
```

### 2. Create Streamlit Cloud Account
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**

### 3. Configure the Deployment
- **Repository:** `YOUR-USERNAME/multi-doc-rag`
- **Branch:** `main`
- **Main file path:** `app.py`

### 4. Add Secrets (Critical! 🔐)
In the Streamlit Cloud dashboard:
1. Click **"Advanced settings"** → **"Secrets"**
2. Paste this (with your real API key):
```
GROQ_API_KEY = "gsk_your_real_api_key_here"
```
3. Click **"Save"**

✅ **Your app is now live!** Share the URL with others.

### 5. Update Your App (via Git Push)
```powershell
# Make changes locally
# ...edit your code...

# Push to GitHub
git add .
git commit -m "Updated RAG chain"
git push

# Streamlit Cloud auto-deploys in ~30 seconds!
```

---

## 📁 Project Structure

```
multi-doc-rag/
├── app.py                    # Main Streamlit UI application
├── requirements.txt          # Python dependencies (24 packages)
├── .env                      # Local API keys (⚠️ never commit!)
├── .gitignore               # Excludes secrets & large files
│
├── .streamlit/
│   └── config.toml          # Streamlit configuration (theme, upload size)
│
├── src/
│   ├── __init__.py
│   ├── chain.py             # RAG chain orchestration with Groq LLM
│   ├── loader.py            # PDF document loading & chunking
│   ├── embedder.py          # HuggingFace embeddings (all-MiniLM-L6-v2)
│   ├── retriever.py         # ChromaDB semantic search
│   └── evaluator.py         # RAGAS quality metrics
│
├── data/                     # Your uploaded PDFs (⚠️ not in git)
├── vectorstore/              # ChromaDB embeddings (⚠️ not in git)
└── README.md                # This file
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `app.py` | Single Streamlit page with upload, Q&A, and evaluation UI |
| `src/chain.py` | Creates LLM chain that combines retriever + Groq API |
| `src/loader.py` | Splits PDFs into chunks for embedding |
| `src/embedder.py` | Converts text to 384-dimensional vectors |
| `src/retriever.py` | Searches ChromaDB for relevant chunks |
| `requirements.txt` | Lists all 24 dependencies (pinned to exact versions) |
| `.streamlit/config.toml` | Streamlit Cloud settings (200MB upload limit, theme) |

---

## 🔑 Environment Variables

### GROQ_API_KEY
Required for LLM inference.

| Environment | Where to Set | How |
|-------------|-------------|-----|
| **Local Development** | `.env` file | `GROQ_API_KEY=gsk_xxx` |
| **Streamlit Cloud** | Dashboard Secrets | UI form in app settings |
| **How it works** | Dual fallback in `src/chain.py` | Tries st.secrets first, then .env |

**Get free API key:**
1. Sign up at [console.groq.com](https://console.groq.com)
2. Generate API key (free tier available)
3. Copy/paste into `.env` or Streamlit Secrets

---

## 💡 How It Works

### Architecture Flow

```
User's PDF
    ↓
[PDF Loader] → Chunks text into 1024-char segments
    ↓
[Embedder] → Converts each chunk to 384-dim vector
    ↓
[ChromaDB] → Stores vectors in SQLite with metadata
    ↓
User's Question
    ↓
[Embedder] → Convert question to 384-dim vector
    ↓
[Semantic Search] → Find 5 most similar chunks (cosine similarity)
    ↓
[RAG Chain] → Format context + question → send to Groq
    ↓
[Groq LLM] → llama-3.1-8b returns answer (very fast!)
    ↓
[Evaluation] → RAGAS scores answer quality
    ↓
User sees answer + quality metrics
```

### Models Used

- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  - Speed: ~5ms per chunk
  - Dimensions: 384
  - Size: 100MB (cached locally)
  - Similarity: Cosine distance

- **Generation Model:** `llama-3.1-8b-instant` (via Groq)
  - Speed: ~1 sec per answer (very fast!)
  - Context: Up to 8K tokens
  - Quality: State-of-the-art for its size
  - Cost: Free on Groq free tier

- **Evaluation Model:** `mixtral-8x7b-32768` (via Groq, optional)
  - Rates answer quality (faithfulness, relevancy, precision)
  - Only runs if you click "Evaluate Answer"

---

## 🧪 Running Evaluations

The RAGAS framework evaluates RAG quality automatically:

```python
# In app.py or via evaluation_results.json
- Faithfulness: Does answer match source documents?
- Answer Relevancy: Is answer relevant to the question?
- Context Precision: Are retrieved chunks actually helpful?
```

Run evaluation to ensure your RAG works well before sharing!

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

---

## ❓ FAQ

**Q: Is this really free?**
A: Yes! Groq free tier + Streamlit Community Cloud = $0/month. No credit card needed after signup.

**Q: Can I use my own PDF?**
A: Absolutely! The app lets you upload any PDF (up to 200MB per Streamlit's limit).

**Q: What if my question requires information from multiple PDFs?**
A: The embedder looks across all documents simultaneously. If the answer exists in any PDF, it will find it!

**Q: Why ChromaDB instead of Pinecone/Weaviate?**
A: It's serverless and free. Pinecone/Weaviate require paid tiers for production. ChromaDB works offline.

**Q: How do I update the app after deploying?**
A: Just push to GitHub (`git push`) and Streamlit Cloud auto-deploys in ~30 seconds. No Docker, no build steps needed.

**Q: What if my API key leaks?**
A: Regenerate it at [console.groq.com](https://console.groq.com) instantly. Never commit `.env` or `.streamlit/secrets.toml` to git.

---

## 🔒 Security Best Practices

⚠️ **NEVER commit these files:**
- `.env` (local secrets)
- `.streamlit/secrets.toml` (Streamlit Cloud secrets)
- `vectorstore/` (your embeddings)
- `data/` (your PDFs)

✅ **These are already in `.gitignore`** — verified in TASK 4

✅ **Streamlit Cloud automatically protects secrets** — they're not visible in code or logs

---

## 📊 Performance Notes

| Operation | Time | Notes |
|-----------|------|-------|
| PDF upload & embedding | ~10-30s per PDF | Depends on length |
| Semantic search | <100ms | Very fast |
| LLM answer generation | ~1-3s | Groq is *very* fast |
| Full Q&A cycle | ~2-5s | Depends on model |

For 100+ page PDFs, patience is required during initial embedding. After that, Q&A is instant!

---

## 🤝 Contributing

Found a bug or have an idea? Fork and submit a PR!

Possible improvements:
- [ ] Support for more file types (DOCX, TXT, images)
- [ ] Multi-language support
- [ ] Chat history persistence
- [ ] Web scraping integration
- [ ] Custom embedding models

---

## **Developer Notes**

- **Run the app locally (Windows PowerShell):**

```powershell
# create and activate venv (if not created)
python -m venv venv
.\venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# run the Streamlit UI
streamlit run app.py
```

- **Run the evaluation script (CLI):**

```powershell
# Evaluate RAG outputs / run automated evaluator
python evaluate_rag.py
```

- **Run unit tests:**

```powershell
# from project root
pip install pytest    # if not already installed
pytest -q
```

- **Rebuild or reset the vectorstore:**

1. Delete the `vectorstore/chroma_db/` directory (or move it) to force reindexing.
2. Re-run the app and upload documents; the app will re-create the chroma DB.

- **Key scripts:**

- `app.py` — Streamlit app (UI, upload, Q&A, evaluation button)
- `evaluate_rag.py` — Batch/CLI evaluator for RAG outputs
- `test_evaluator.py` — Unit tests for the evaluator logic

- **Where embeddings are stored:**

The ChromaDB SQLite files live in `vectorstore/chroma_db/`. Don’t commit them.

---

## 📝 License

MIT License - Use freely for personal and commercial projects.

---

## 🙋 Questions?

Open an issue on GitHub or check out:
- [LangChain Docs](https://python.langchain.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [Groq Console](https://console.groq.com)

---

*Last updated: May 7, 2026*
