"""
Retriever Module

Simple retriever that fetches top-4 most similar chunks from vectorstore.
Uses cosine similarity search.
"""

from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma


def create_retriever(vectorstore: Chroma, k: int = 4) -> BaseRetriever:
    """
    Create a retriever from a Chroma vectorstore.
    
    Args:
        vectorstore: Chroma vectorstore object
        k: Number of chunks to retrieve (default: 4)
    
    Returns:
        LangChain retriever object
    
    Example:
        vectorstore = load_vectorstore(embeddings)
        retriever = create_retriever(vectorstore, k=4)
        chunks = retriever.invoke("What is machine learning?")
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def format_retrieved_chunks(chunks: list) -> str:
    """
    Format retrieved chunks with source and page info for prompt.
    
    Args:
        chunks: List of retrieved Document objects
    
    Returns:
        Formatted string with metadata and content
    """
    formatted = ""
    
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "Unknown source")
        page = chunk.metadata.get("page", "Unknown page")
        content = chunk.page_content
        
        formatted += f"\n[Chunk {i}]\n"
        formatted += f"Source: {source} (Page {page})\n"
        formatted += f"Content: {content}\n"
        formatted += "-" * 60 + "\n"
    
    return formatted


if __name__ == "__main__":
    print("Retriever module loaded successfully")
