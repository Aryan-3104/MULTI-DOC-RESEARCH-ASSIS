"""
Embedding and Vectorstore Module

Creates and manages ChromaDB vectorstore with HuggingFace embeddings (all-MiniLM-L6-v2).
Persists vectorstore to disk to avoid re-embedding.
Optimized for 8GB RAM with batch_size=16.
"""

import os
import shutil
import time
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Vectorstore configuration
VECTORSTORE_DIR = "vectorstore"
PERSIST_DIRECTORY = os.path.join(VECTORSTORE_DIR, "chroma_db")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16  # Optimized for 8GB RAM


def load_embeddings():
    """
    Load HuggingFace embeddings model.
    
    Uses all-MiniLM-L6-v2: lightweight, CPU-friendly, no API calls.
    Model is downloaded and cached locally on first run (~100MB).
    
    Returns:
        HuggingFaceEmbeddings object
    """
    print(f"Loading embeddings model: {MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"  # Use CPU to conserve GPU/RAM
        },
        encode_kwargs={
            "batch_size": BATCH_SIZE,  # Process 16 chunks at a time
            "normalize_embeddings": True  # L2 normalization
        }
    )
    print("✓ Embeddings model loaded")
    return embeddings


def create_vectorstore(chunks: list, embeddings) -> Chroma:
    """
    Create a new ChromaDB vectorstore from chunks.
    
    Args:
        chunks: List of Document objects with metadata
        embeddings: HuggingFaceEmbeddings object
    
    Returns:
        Chroma vectorstore object
    """
    print(f"\nCreating vectorstore with {len(chunks)} chunks...")
    
    # Create vectorstore and persist to disk
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    print(f"✓ Vectorstore created and persisted to {PERSIST_DIRECTORY}")
    return vectorstore


def load_vectorstore(embeddings) -> Chroma:
    """
    Load existing ChromaDB vectorstore from disk.
    
    Checks if vectorstore exists. If not, raises error (use create_or_load_vectorstore instead).
    
    Args:
        embeddings: HuggingFaceEmbeddings object
    
    Returns:
        Chroma vectorstore object
        
    Raises:
        FileNotFoundError: If vectorstore directory doesn't exist
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Vectorstore not found at {PERSIST_DIRECTORY}")
    
    print(f"Loading vectorstore from {PERSIST_DIRECTORY}...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    collection_count = vectorstore._collection.count()
    print(f"✓ Vectorstore loaded with {collection_count} chunks")
    return vectorstore


def create_or_load_vectorstore(chunks: list = None, embeddings=None, force_recreate: bool = False) -> Chroma:
    """
    Load existing vectorstore from disk, or create new one from chunks.
    
    Args:
        chunks: List of Document objects (required if creating new)
        embeddings: HuggingFaceEmbeddings object (required if creating new)
        force_recreate: If True, delete existing vectorstore and recreate (OPTIONAL - for backward compatibility)
    
    Returns:
        Chroma vectorstore object
    """
    
    # If force_recreate requested but vectorstore already deleted, that's fine
    if force_recreate and os.path.exists(VECTORSTORE_DIR):
        print("⚠️  force_recreate=True, attempting to delete old vectorstore...")
        delete_success = delete_vectorstore()
        if not delete_success:
            print("⚠️  Delete returned False, but proceeding anyway...")
    
    # Load existing vectorstore
    if os.path.exists(PERSIST_DIRECTORY):
        if embeddings is None:
            embeddings = load_embeddings()
        return load_vectorstore(embeddings)
    
    # Create new vectorstore
    if chunks is None or embeddings is None:
        raise ValueError(
            "chunks and embeddings required to create new vectorstore. "
            "Use: create_or_load_vectorstore(chunks, embeddings)"
        )
    
    return create_vectorstore(chunks, embeddings)


def cleanup_vectorstore_connections():
    """
    Close and cleanup all Chroma database connections, embeddings, and file handles.
    
    This is critical for Windows file locking issues - all references must be explicitly
    cleared and garbage collected before attempting deletion. Call this BEFORE delete_vectorstore().
    
    Returns:
        None
    """
    # Import gc to force cleanup
    import gc
    # Import sys to access modules
    import sys
    
    print("🔌 Cleaning up connections and removing module references...")
    
    # Step 1: Try to explicitly close ChromaDB client if it exists
    print("  Step 1: Closing ChromaDB client...")
    try:
        import chromadb
        try:
            # Try multiple ways to close the client
            if hasattr(chromadb, '_ClientSingleton'):
                chromadb._ClientSingleton = None
            # Try to reset the global client
            chromadb._cached_client = None
        except Exception:
            pass
        
        try:
            # Try to get and close the client directly
            client = chromadb.get_client()
            if hasattr(client, 'close'):
                client.close()
            if hasattr(client, '_client'):
                if hasattr(client._client, 'close'):
                    client._client.close()
        except Exception:
            pass
        print("    ✓ ChromaDB client cleanup attempted")
    except Exception as e:
        print(f"    ⚠️  ChromaDB not available: {e}")
    
    # Step 2: Unload ChromaDB and related modules from sys.modules
    print("  Step 2: Unloading ChromaDB modules...")
    modules_to_remove = [key for key in sys.modules.keys() if 'chroma' in key.lower()]
    for module_name in modules_to_remove:
        try:
            del sys.modules[module_name]
        except Exception:
            pass
    if modules_to_remove:
        print(f"    ✓ Unloaded {len(modules_to_remove)} ChromaDB-related modules")
    
    # Step 3: Force aggressive garbage collection
    print("  Step 3: Forcing garbage collection...")
    for i in range(10):  # 10 passes instead of 5
        gc.collect()
    print("    ✓ Garbage collection completed (10 passes)")
    
    # Step 4: Sleep longer to ensure OS releases all locks
    print("  Step 4: Waiting for OS to release file locks...")
    import time
    time.sleep(2)  # Increased from 1.5 to 2 seconds
    print("    ✓ Cleanup complete")


def delete_vectorstore() -> bool:
    """
    Delete the vectorstore directory using Windows-compatible approach.
    
    Strategy: Rename the directory first (works even with locks), then delete it.
    If rename succeeds, return True immediately - the old vectorstore is isolated.
    
    Returns:
        True if successfully isolated/deleted, False only if rename fails completely
    """
    # Import datetime for unique timestamps
    from datetime import datetime
    
    # Check if the vectorstore directory exists on disk at the expected location
    if not os.path.exists(VECTORSTORE_DIR):
        # Vectorstore directory was already deleted or never existed, return success
        print(f"ℹ No vectorstore found at {VECTORSTORE_DIR}")
        return True
    
    print(f"🗑️  Deleting vectorstore at {VECTORSTORE_DIR}...")
    
    try:
        # PRIMARY STRATEGY: Rename the directory first
        # This ALWAYS works on Windows even if files are locked, because Windows allows
        # renaming directories with open file handles
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_dir_name = f"vectorstore_old_{timestamp}"
        
        print(f"  Step 1: Renaming directory to {temp_dir_name}...")
        try:
            os.rename(VECTORSTORE_DIR, temp_dir_name)
            print(f"  ✓ Successfully renamed!")
            print(f"  ✓ Original 'vectorstore' is now isolated")
            
            # Try to clean it up right now (best effort, don't fail if it doesn't work)
            print(f"  Step 2: Cleaning up renamed directory...")
            try:
                shutil.rmtree(temp_dir_name, ignore_errors=True)
                print(f"  ✓ Renamed directory cleaned up")
            except Exception:
                print(f"  ⚠️  Will clean up on next startup")
            
            print(f"✅ Vectorstore successfully cleared!")
            return True
        
        except OSError as e:
            # Rename failed - try direct deletion
            print(f"  Rename failed: {e}")
            print(f"  Attempting direct deletion...")
            
            try:
                shutil.rmtree(VECTORSTORE_DIR, ignore_errors=True)
                time.sleep(0.3)
                if not os.path.exists(VECTORSTORE_DIR):
                    print(f"✅ Vectorstore deleted successfully")
                    return True
            except Exception:
                pass
            
            print(f"❌ Could not delete vectorstore")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def cleanup_old_vectorstores():
    """
    Clean up old renamed vectorstore directories from previous deletion attempts.
    
    This is called on app startup to clean up any vectorstore_old_* directories
    that couldn't be deleted during active sessions. By the time the app restarts,
    all file locks will be released, allowing deletion to succeed.
    
    Returns:
        None (silently cleans up in background)
    """
    import glob
    
    # Find all old vectorstore directories
    old_dirs = glob.glob("vectorstore_old_*")
    
    if old_dirs:
        print(f"🧹 Cleaning up {len(old_dirs)} old vectorstore directory/ies from previous sessions...")
        for old_dir in old_dirs:
            try:
                shutil.rmtree(old_dir, ignore_errors=True)
                print(f"  ✓ Deleted {old_dir}")
            except Exception as e:
                print(f"  ⚠️  Could not delete {old_dir}: {e}")




def get_vectorstore_info(vectorstore: Chroma) -> dict:
    """
    Get information about the vectorstore.
    
    Args:
        vectorstore: Chroma vectorstore object
    
    Returns:
        Dictionary with vectorstore stats
    """
    collection_count = vectorstore._collection.count()
    
    # Get all documents to analyze metadata
    docs = vectorstore._collection.get(include=[])["ids"]
    
    sources = set()
    max_page = 0
    
    if docs:
        all_metadata = vectorstore._collection.get(include=["metadatas"])["metadatas"]
        for meta in all_metadata:
            if "source" in meta:
                sources.add(meta["source"])
            if "page" in meta:
                max_page = max(max_page, meta["page"])
    
    return {
        "total_chunks": collection_count,
        "unique_sources": len(sources),
        "sources": list(sources),
        "max_page": max_page,
        "persist_directory": PERSIST_DIRECTORY
    }

def get_unique_sources_count(vectorstore) -> int:
    """
    Count how many unique PDF sources are in the vectorstore.
    Used to dynamically set k for retrieval.
    
    Returns:
        Number of unique source documents
    """
    try:
        all_metadata = vectorstore._collection.get(
            include=["metadatas"]
        )["metadatas"]
        
        sources = set()
        for meta in all_metadata:
            if "source" in meta:
                sources.add(meta["source"])
        
        print(f"Unique sources found: {len(sources)} → {list(sources)}")
        return len(sources)
    
    except Exception as e:
        print(f"Could not count sources: {e}. Defaulting to k=3")
        return 1  # fallback
    
    
if __name__ == "__main__":
    # Test embeddings load
    embeddings = load_embeddings()
    print(f"\nEmbedding dimension: {len(embeddings.embed_query('test'))}")
    
    # If vectorstore exists, load it
    if os.path.exists(PERSIST_DIRECTORY):
        vectorstore = load_vectorstore(embeddings)
        info = get_vectorstore_info(vectorstore)
        print(f"\nVectorstore info: {info}")
    else:
        print(f"\nNo vectorstore found at {PERSIST_DIRECTORY}")
        print("Create one with: create_or_load_vectorstore(chunks, embeddings)")