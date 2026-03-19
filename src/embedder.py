"""
Embedding and Vectorstore Module

Creates and manages ChromaDB vectorstore with HuggingFace embeddings (all-MiniLM-L6-v2).
Persists vectorstore to disk to avoid re-embedding.
Optimized for 8GB RAM with batch_size=16.
"""

import os
import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
        force_recreate: If True, delete existing vectorstore and recreate
    
    Returns:
        Chroma vectorstore object
    
    Example:
        # Load existing vectorstore (fast)
        vectorstore = create_or_load_vectorstore()
        
        # Create new vectorstore from chunks
        chunks = load_and_chunk_pdfs()
        embeddings = load_embeddings()
        vectorstore = create_or_load_vectorstore(chunks, embeddings)
        
        # Force recreate even if exists
        vectorstore = create_or_load_vectorstore(chunks, embeddings, force_recreate=True)
    """
    
    # Force recreate: delete existing vectorstore
    if force_recreate and os.path.exists(VECTORSTORE_DIR):
        print(f"Deleting existing vectorstore at {VECTORSTORE_DIR}...")
        shutil.rmtree(VECTORSTORE_DIR)
    
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


def delete_vectorstore() -> None:
    """
    Delete the persisted vectorstore directory.
    
    Use with caution - this removes all embeddings and requires re-embedding from scratch.
    """
    if os.path.exists(VECTORSTORE_DIR):
        print(f"Deleting vectorstore at {VECTORSTORE_DIR}...")
        shutil.rmtree(VECTORSTORE_DIR)
        print("✓ Vectorstore deleted")
    else:
        print(f"No vectorstore found at {VECTORSTORE_DIR}")


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
