"""
Streamlit UI for Multi-Document RAG Application

Upload PDFs, ask questions, get answers with source citations.
"""

import streamlit as st
from pathlib import Path

from src.loader import load_and_chunk_pdfs
from src.embedder import create_or_load_vectorstore, load_embeddings, delete_vectorstore
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
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ============================================================================
# PAGE HEADER
# ============================================================================

st.title("📚 Multi-Document RAG Assistant")
st.markdown("Upload PDFs and ask questions about their content!")


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
            success = delete_vectorstore()
            if success:
                st.session_state.documents_processed = False
                st.session_state.rag_chain = None
                st.session_state.vectorstore = None
                st.success("✅ Cache cleared! You can now upload new documents.")
            else:
                st.error("❌ Failed to clear cache. Try closing the app and deleting the 'vectorstore' folder manually.")
    
    # Process documents when button clicked
    if process_button:
        if not uploaded_files:
            st.error("❌ Please upload at least one PDF file first")
        else:
            with st.spinner("📚 Processing documents..."):
                try:
                    # Create data directory and save files
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        save_path = data_dir / uploaded_file.name
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load and chunk PDFs
                    documents = load_and_chunk_pdfs(str(data_dir))
                    
                    if not documents:
                        st.error("❌ No content extracted from PDFs")
                    else:
                        st.write(f"✓ Loaded {len(documents)} chunks from PDFs")
                        
                        # Load embeddings
                        st.write("Loading embeddings model...")
                        embeddings = load_embeddings()
                        
                        # Create vectorstore
                        st.write("Building vectorstore...")
                        vectorstore = create_or_load_vectorstore(
                            chunks=documents,
                            embeddings=embeddings,
                            force_recreate=True
                        )
                        st.session_state.vectorstore = vectorstore
                        
                        # Create retriever and RAG chain
                        st.write("Building RAG chain...")
                        retriever = create_retriever(vectorstore)
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
                    
                    # Special handling for file lock errors
                    if "WinError 32" in error_msg or "being used by another process" in error_msg:
                        st.warning("""
                        **File Lock Error (Windows)**
                        
                        The vectorstore is locked. Try these steps:
                        1. Click the "🗑️ Clear Vectorstore Cache" button above
                        2. Close the browser tab completely
                        3. Wait 5 seconds
                        4. Refresh and try again
                        5. If still failing, manually delete the 'vectorstore' folder
                        """)
                    
                    st.session_state.documents_processed = False
    
    # Show status
    st.divider()
    if st.session_state.documents_processed:
        st.success("✓ Documents loaded and ready!")
    else:
        st.warning("No documents processed yet")


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.documents_processed:
    # Show getting started message
    st.info(
        "👈 **Get started:** Upload PDF files in the sidebar and click 'Process Documents' to begin."
    )
    st.markdown("""
    ### How to use:
    1. **Upload** PDF files using the sidebar uploader
    2. **Process** them by clicking the button
    3. **Ask** questions about your documents
    4. **Get** answers with source citations
    """)
else:
    # Show ready state and question input
    st.success("✓ Ready to answer questions about your documents!")
    
    st.divider()
    st.header("❓ Ask a Question")
    
    # Question input
    user_question = st.text_input(
        "What would you like to know?",
        placeholder="E.g., What are the main points in these documents?",
        help="Type your question and press Enter"
    )
    
    # Process question
    if user_question:
        with st.spinner("🤔 Searching documents and generating answer..."):
            try:
                # Invoke the RAG chain
                response = invoke_chain(st.session_state.rag_chain, user_question)
                
                # Display answer
                st.markdown("### 📖 Answer")
                
                # Handle both string and dict responses
                if isinstance(response, dict):
                    answer_text = response.get("answer", response.get("result", str(response)))
                    source_docs = response.get("source_documents", [])
                else:
                    answer_text = str(response)
                    source_docs = []
                
                st.markdown(answer_text)
                
                # Display sources if available
                if source_docs:
                    st.divider()
                    st.markdown("### 📝 Source Documents")
                    
                    for i, doc in enumerate(source_docs, 1):
                        with st.expander(f"Source {i}"):
                            # Get metadata
                            source_name = doc.metadata.get("source", "Unknown")
                            page = doc.metadata.get("page", "N/A")
                            
                            st.write(f"**File:** {source_name}")
                            st.write(f"**Page:** {page}")
                            st.write(f"**Excerpt:**")
                            st.text(doc.page_content)
                
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
                st.markdown("""
                **Troubleshooting:**
                - Ensure your `GROQ_API_KEY` is set in `.env`
                - Check that documents have been processed successfully
                - Try rephrasing your question
                """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "Multi-Document RAG Application | Powered by LangChain, ChromaDB, and Groq"
    "</div>",
    unsafe_allow_html=True
)
