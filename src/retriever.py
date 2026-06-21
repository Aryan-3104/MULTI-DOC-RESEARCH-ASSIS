"""
Retriever Module

Dynamically sets k based on number of uploaded documents.
Formula: k = num_sources * 3, bounded between 3 and 15.
"""

from langchain_community.vectorstores import Chroma
from src.embedder import get_unique_sources_count


def create_retriever(vectorstore):
    """
    Dynamically sets k based on number of uploaded documents.
    
    Formula: k = num_sources * 3
    Min k = 3, Max k = 15
    
    Args:
        vectorstore: Chroma vectorstore object
    
    Returns:
        LangChain retriever object
    """
    # Count unique documents in vectorstore
    num_sources = get_unique_sources_count(vectorstore)
    
    # 3 chunks per document, bounded between 3 and 12
    k = max(3, min(num_sources * 3, 12))
    
    print(f"Documents detected: {num_sources} → Setting k={k}")
    
    # MMR (Maximal Marginal Relevance) reduces redundant chunks,
    # which directly improves RAGAS context_precision score.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 3, "lambda_mult": 0.7}
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