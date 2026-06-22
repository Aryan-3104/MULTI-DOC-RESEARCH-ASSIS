"""
Streamlit UI for Multi-Document RAG Application

Upload PDFs, ask questions, get answers with source citations.
"""

import nest_asyncio
nest_asyncio.apply = lambda: None

import streamlit as st
from pathlib import Path

from src.loader import load_and_chunk_pdfs
from src.embedder import create_or_load_vectorstore, load_embeddings, delete_vectorstore, get_vectorstore_info
from src.retriever import create_retriever
from src.chain import build_rag_chain, invoke_chain




# Streamlit page configuration
st.set_page_config(
    page_title="Multi-Document RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling injection
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    /* Premium visual fonts */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Elegant Title Header styling */
    h1, .main-title {
        font-family: 'Outfit', sans-serif !important;
        background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar premium dark background adjustment */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* Styled action buttons with linear gradients and lift on hover */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2), 0 2px 4px -2px rgba(99, 102, 241, 0.2) !important;
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.35), 0 4px 6px -4px rgba(99, 102, 241, 0.35) !important;
        border-color: transparent !important;
        color: #FFFFFF !important;
    }
    
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Clear Cache secondary button special styling */
    div.stButton > button[key="clear_cache_btn"] {
        background: #1E293B !important;
        color: #94A3B8 !important;
        border: 1px solid #334155 !important;
        box-shadow: none !important;
    }
    
    div.stButton > button[key="clear_cache_btn"]:hover {
        background: #334155 !important;
        color: #F8FAFC !important;
        transform: none !important;
    }
    
    /* Dashboard KPI metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        transition: border-color 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #475569 !important;
    }
    
    div[data-testid="metric-container"] label {
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Custom spacing and styling for chat messages */
    .stChatMessage {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
        padding: 0.8rem 1rem !important;
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #2E2A6E !important;
        border-color: #4338CA !important;
    }
    
    /* File uploader styling and labels */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #1E293B !important;
        border: 1px dashed #475569 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #F8FAFC !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Quick Start Guide Card Hover Effects */
    .step-card {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .step-card:hover {
        transform: translateY(-5px) !important;
        background: rgba(30, 41, 59, 0.7) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 12px 24px -10px rgba(99, 102, 241, 0.25) !important;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0F172A;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None  # Cache embeddings to avoid PyO3 reload
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# Load embeddings model at app startup (only once)
# This ensures embeddings are always available when needed
if st.session_state.embeddings is None:
    with st.spinner("Loading embeddings model on startup..."):
        st.session_state.embeddings = load_embeddings()


# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.05) 100%);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2.2rem;
            display: flex;
            align-items: center;
            gap: 1.25rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(8px);">
    <div style="background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
                width: 54px;
                height: 54px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.8rem;
                box-shadow: 0 8px 16px rgba(99, 102, 241, 0.3);">
        📚
    </div>
    <div>
        <h1 style="margin: 0; font-size: 2rem; font-family: 'Outfit', sans-serif; font-weight: 800; background: linear-gradient(135deg, #F8FAFC 0%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;">
            Multi-Doc RAG AI Assistant
        </h1>
        <p style="margin: 0.2rem 0 0 0; color: #94A3B8; font-size: 0.95rem; font-family: 'Inter', sans-serif;">
            Analyze and extract intelligence from all your PDF documents simultaneously.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR: PDF Upload & Processing
# ============================================================================

with st.sidebar:
    st.header("📤 Upload Documents")
    st.write("Upload one or more PDF files to create a searchable knowledge base.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )
    
    # Show uploaded file count
    if uploaded_files:
        st.write(f"**Files selected:** {len(uploaded_files)}")
        for file in uploaded_files:
            st.caption(f"📄 {file.name}")
    
    # Process button
    process_button = st.button(
        "🔄 Process Documents",
        type="primary",
        use_container_width=True,
        key="process_btn"
    )
    
    # Clear cache button (helpful for file lock issues)
    st.write("")  # Spacing
    clear_cache = st.button(
        "🗑️ Clear Vectorstore Cache",
        use_container_width=True,
        key="clear_cache_btn",
        help="Delete cached embeddings. Use if you get file lock errors."
    )
    
    if clear_cache:
        with st.spinner("Clearing vectorstore cache..."):
            # Clear session state to release all objects from memory
            st.session_state.documents_processed = False
            st.session_state.rag_chain = None
            st.session_state.vectorstore = None
            st.session_state.retriever = None
            st.session_state.messages = []
            # Don't clear embeddings - we'll reuse it to avoid PyO3 reinitialization
            
            # Cleanup ChromaDB connections before deletion
            cleanup_vectorstore_connections()
            
            # Delete the vectorstore directory (renames it out of the way)
            success = delete_vectorstore()
            
            if success:
                st.success("✅ Vectorstore cleared successfully!")
                st.info("📝 You can now upload new PDFs and process them.")
            else:
                st.error("❌ Failed to clear cache. Please restart the app manually.")
    
    # Process documents when button clicked
    if process_button:
        if not uploaded_files:
            st.error("❌ Please upload at least one PDF file first")
        else:
            with st.spinner("📚 Processing documents..."):
                try:
                    # ===== CRITICAL FIX: Don't reload ChromaDB unnecessarily =====
                    st.write("🧹 Step 1: Cleaning up old vectorstore from memory...")
                    
                    # Release vectorstore and retriever objects
                    st.session_state.vectorstore = None
                    st.session_state.retriever = None
                    st.session_state.rag_chain = None
                    st.session_state.messages = []
                    
                    # Embeddings are already loaded at app startup - use cached version
                    st.write("✓ Using cached embeddings model")
                    
                    # Force garbage collection to release ChromaDB file locks
                    import gc
                    gc.collect()
                    
                    # Wait for locks to release on Windows
                    import time
                    time.sleep(1.0)
                    # ===================================================================

                    
                    # ===== Delete vectorstore folder =====
                    st.write("🗑️  Step 2: Deleting old vectorstore folder from disk...")
                    
                    try:
                        delete_vectorstore()
                    except Exception as _del_err:
                        st.warning(f"⚠️  Could not fully delete old vectorstore: {_del_err}. Proceeding anyway...")
                    # ======================================
                    
                    # Create data directory and save files
                    st.write("📁 Step 3: Saving uploaded PDFs...")
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    
                    # Delete old PDFs to prevent stale sources
                    for old_file in data_dir.glob("*.pdf"):
                        old_file.unlink()
                    
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        save_path = data_dir / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load and chunk PDFs
                    st.write("📄 Step 4: Loading and chunking PDFs...")
                    documents = load_and_chunk_pdfs(str(data_dir))
                    
                    if not documents:
                        st.error("❌ No content extracted from PDFs")
                    else:
                        st.write(f"✓ Loaded {len(documents)} chunks from PDFs")
                        
                        # Use cached embeddings to avoid PyO3 reinitialization
                        st.write("🗂️  Step 5: Creating fresh vectorstore...")
                        vectorstore = create_or_load_vectorstore(
                            chunks=documents,
                            embeddings=st.session_state.embeddings,
                            force_recreate=False
                        )
                        st.session_state.vectorstore = vectorstore
                        
                        # Create retriever and RAG chain
                        st.write("🔗 Step 6: Building RAG chain...")
                        retriever = create_retriever(vectorstore)
                        st.session_state.retriever = retriever
                        rag_chain = build_rag_chain(retriever)
                        st.session_state.rag_chain = rag_chain
                        st.session_state.documents_processed = True
                        
                        st.success(
                            f"✅ Successfully processed {len(uploaded_files)} file(s) "
                            f"with {len(documents)} chunks!"
                        )
                        
                except Exception as e:
                    error_msg = str(e)
                    st.error(f"❌ Error processing documents: {error_msg}")
                    import traceback
                    st.write(traceback.format_exc())
                    st.session_state.documents_processed = False
    
    # Show status and dashboard stats cards
    st.divider()
    if st.session_state.documents_processed:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 10px; padding: 0.8rem; margin: 1rem 0; display: flex; align-items: center; gap: 0.5rem; color: #34D399; font-weight: 500; font-size: 0.9rem; font-family: 'Outfit', sans-serif;">
            <span style="font-size: 1.15rem;">✅</span> Database Active & Ready
        </div>
        """, unsafe_allow_html=True)
        
        # Display dynamic stats cards inside the sidebar
        if st.session_state.vectorstore:
            try:
                info = get_vectorstore_info(st.session_state.vectorstore)
                st.markdown(f"""
                <div style="margin-top: 1.25rem;">
                    <h4 style="font-family: 'Outfit', sans-serif; color: #F8FAFC; font-size: 0.95rem; margin-bottom: 0.75rem; font-weight: 600;">📊 Database Metrics</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;">
                        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 0.75rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                            <div style="font-size: 1.35rem; font-weight: 700; color: #6366F1; font-family: 'Outfit', sans-serif;">{info.get("unique_sources", 0)}</div>
                            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem; font-weight: 500;">Unique Files</div>
                        </div>
                        <div style="background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 0.75rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.15);">
                            <div style="font-size: 1.35rem; font-weight: 700; color: #C084FC; font-family: 'Outfit', sans-serif;">{info.get("total_chunks", 0)}</div>
                            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem; font-weight: 500;">Vector Chunks</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 10px; padding: 0.8rem; margin: 1rem 0; display: flex; align-items: center; gap: 0.5rem; color: #FBBF24; font-weight: 500; font-size: 0.9rem; font-family: 'Outfit', sans-serif;">
            <span style="font-size: 1.15rem;">⚠️</span> No documents processed yet
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.documents_processed:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%); 
                border: 1px solid rgba(255, 255, 255, 0.05); 
                border-radius: 20px; 
                padding: 2.2rem; 
                margin-top: 0.5rem;
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(12px);">
        <h2 style="font-family: 'Outfit', sans-serif; font-weight: 700; color: #F8FAFC; margin-top: 0; margin-bottom: 1.5rem; font-size: 1.7rem; display: flex; align-items: center; gap: 0.5rem; letter-spacing: -0.3px;">
            🚀 Welcome & Quick Start Guide
        </h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.25rem; margin-bottom: 2rem;">
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 1.5rem; cursor: default;" class="step-card">
                <div style="background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; margin-bottom: 1rem; font-family: 'Outfit', sans-serif; font-size: 1.05rem;">1</div>
                <h4 style="font-family: 'Outfit', sans-serif; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem; margin-top: 0; font-size: 1.1rem;">Upload PDFs</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45; margin: 0;">Select one or multiple PDF documents in the sidebar uploader.</p>
            </div>
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 1.5rem; cursor: default;" class="step-card">
                <div style="background: linear-gradient(135deg, #EC4899 0%, #D946EF 100%); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; margin-bottom: 1rem; font-family: 'Outfit', sans-serif; font-size: 1.05rem;">2</div>
                <h4 style="font-family: 'Outfit', sans-serif; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem; margin-top: 0; font-size: 1.1rem;">Process Files</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45; margin: 0;">Click 'Process Documents' to extract and build the local embeddings index.</p>
            </div>
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 1.5rem; cursor: default;" class="step-card">
                <div style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; margin-bottom: 1rem; font-family: 'Outfit', sans-serif; font-size: 1.05rem;">3</div>
                <h4 style="font-family: 'Outfit', sans-serif; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem; margin-top: 0; font-size: 1.1rem;">Ask Questions</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45; margin: 0;">Use the interactive chat input at the bottom to query all sources at once.</p>
            </div>
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 16px; padding: 1.5rem; cursor: default;" class="step-card">
                <div style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-weight: 700; color: white; margin-bottom: 1rem; font-family: 'Outfit', sans-serif; font-size: 1.05rem;">4</div>
                <h4 style="font-family: 'Outfit', sans-serif; font-weight: 600; color: #F8FAFC; margin-bottom: 0.5rem; margin-top: 0; font-size: 1.1rem;">Inspect Sources</h4>
                <p style="color: #94A3B8; font-size: 0.88rem; line-height: 1.45; margin: 0;">Expand detailed source snippets with filename and page numbers under answers.</p>
            </div>
        </div>
        <div style="background: rgba(99, 102, 241, 0.08); border-left: 4px solid #6366F1; border-radius: 10px; padding: 1.1rem; display: flex; align-items: center; gap: 0.75rem; border-top: 1px solid rgba(99, 102, 241, 0.1); border-right: 1px solid rgba(99, 102, 241, 0.1); border-bottom: 1px solid rgba(99, 102, 241, 0.1);">
            <span style="font-size: 1.4rem; filter: drop-shadow(0 2px 4px rgba(99, 102, 241, 0.2));">👈</span>
            <div style="color: #E2E8F0; font-size: 0.92rem; line-height: 1.4; font-family: 'Inter', sans-serif;">
                <strong>Ready to start?</strong> Simply upload one or more PDFs in the left sidebar uploader, then click <strong>Process Documents</strong>.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display scrollable chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📝 View Source Citations"):
                    for idx, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page']})")
                        st.caption(src['content'])
                        if idx < len(message["sources"]):
                            st.divider()

    # Chat input at the bottom
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🤔 Searching documents and generating answer..."):
                try:
                    # Retrieve source documents to save in history
                    source_docs = []
                    if st.session_state.retriever:
                        source_docs = st.session_state.retriever.invoke(prompt)
                    
                    # Generate answer using RAG chain
                    response = invoke_chain(st.session_state.rag_chain, prompt)
                    answer_text = str(response)
                    
                    # Display the text
                    message_placeholder.markdown(answer_text)
                    
                    # Format sources metadata list
                    sources_data = [
                        {
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "N/A"),
                            "content": doc.page_content
                        } for doc in source_docs
                    ]
                    
                    # Show sources expander right below the message
                    if sources_data:
                        with st.expander("📝 View Source Citations"):
                            for idx, src in enumerate(sources_data, 1):
                                st.markdown(f"**Source {idx}:** `{src['source']}` (Page {src['page']})")
                                st.caption(src['content'])
                                if idx < len(sources_data):
                                    st.divider()
                    
                    # Save assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources_data
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.markdown("""
                    **Troubleshooting:**
                    - Ensure your `GROQ_API_KEY` is set correctly in `.env` or Streamlit Secrets.
                    - Try clearing the vectorstore cache and re-processing documents.
                    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85em; font-weight: 500;'>"
    "Multi-Document RAG Application | Powered by LangChain, ChromaDB, and Groq"
    "</div>",
    unsafe_allow_html=True
)