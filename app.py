"""
Streamlit UI for Multi-Document RAG Application

Upload PDFs, ask questions, get answers with source citations.
"""

import os
import tempfile
import streamlit as st
from pathlib import Path

from src.loader import load_and_chunk_pdfs
from src.embedder import load_embeddings, create_or_load_vectorstore
from src.retriever import create_retriever
from src.chain import build_rag_chain, invoke_chain


# Streamlit page configuration
st.set_page_config(
    page_title="Multi-Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# ============================================================================
# SIDEBAR: PDF Upload & Processing
# ============================================================================

st.sidebar.title("📄 Document Upload")
st.sidebar.write("Upload PDF files to create a searchable knowledge base.")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files"
)

# Show uploaded files
if uploaded_files:
    st.sidebar.write(f"**Files selected:** {len(uploaded_files)}")
    for file in uploaded_files:
        st.sidebar.write(f"  • {file.name}")
else:
    st.sidebar.info("No files uploaded yet")

# Process button
process_button = st.sidebar.button(
    "🚀 Process Documents",
    type="primary",
    use_container_width=True
)

# Process documents when button clicked
if process_button:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file first")
    else:
        with st.spinner("Processing documents..."):
            try:
                # Create temporary directory for PDFs
                data_dir = "data"
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                
                # Save uploaded files to data directory
                st.sidebar.write(f"Saving {len(uploaded_files)} files...")
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(data_dir, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Saved {len(uploaded_files)} files")
                
                # Load and chunk PDFs
                st.sidebar.write("Loading and chunking PDFs...")
                chunks = load_and_chunk_pdfs(data_dir=data_dir)
                
                if not chunks:
                    st.sidebar.error("No content extracted from PDFs")
                else:
                    st.sidebar.success(f"Created {len(chunks)} chunks")
                    
                    # Load embeddings
                    st.sidebar.write("Loading embeddings model...")
                    embeddings = load_embeddings()
                    st.session_state.embeddings = embeddings
                    
                    # Create or load vectorstore
                    st.sidebar.write("Building vectorstore...")
                    vectorstore = create_or_load_vectorstore(
                        chunks=chunks,
                        embeddings=embeddings,
                        force_recreate=True
                    )
                    st.session_state.vectorstore = vectorstore
                    
                    # Create retriever and chain
                    st.sidebar.write("Building RAG chain...")
                    retriever = create_retriever(vectorstore, k=4)
                    chain = build_rag_chain(retriever)
                    st.session_state.chain = chain
                    st.session_state.documents_loaded = True
                    
                    st.sidebar.success("✓ Ready to answer questions!")
                    
            except Exception as e:
                st.sidebar.error(f"Error processing documents: {e}")


# Show status in sidebar
st.sidebar.divider()
if st.session_state.documents_loaded:
    st.sidebar.success("✓ Documents loaded and ready")
    if st.session_state.vectorstore:
        count = st.session_state.vectorstore._collection.count()
        st.sidebar.write(f"💾 Vectorstore: {count} chunks")
else:
    st.sidebar.warning("⚠ No documents loaded yet")


# ============================================================================
# MAIN: Question & Answer
# ============================================================================

st.title("📚 Multi-Document Research Assistant")
st.write("Ask questions about your uploaded documents. Answers include source citations.")

# Check if documents are loaded
if not st.session_state.documents_loaded:
    st.info(
        "👈 **Getting Started:**\n\n"
        "1. Upload PDF files in the sidebar\n"
        "2. Click 'Process Documents'\n"
        "3. Ask questions about your documents"
    )
else:
    st.success("✓ Documents loaded and ready for questions")
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="E.g., What is the main topic of these documents?",
        help="Type your question and press Enter to get an answer"
    )
    
    # Handle question submission
    if question:
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Invoke the RAG chain
                answer = invoke_chain(st.session_state.chain, question)
                
                # Display answer
                st.subheader("Answer")
                st.markdown(answer)
                
                # Divider
                st.divider()
                
                # Show helpful message
                st.info("💡 Tip: The answer above includes citations showing which documents were used.")
                
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                st.write("Please check that:")
                st.write("- Your GROQ_API_KEY is set in .env")
                st.write("- Documents have been processed successfully")


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    Multi-Document RAG Application | Powered by LangChain, ChromaDB, and Groq
    </div>
    """,
    unsafe_allow_html=True
)
