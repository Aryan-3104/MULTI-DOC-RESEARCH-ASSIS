"""
PDF Loader and Chunking Module

Loads PDFs from the data/ folder and chunks them using RecursiveCharacterTextSplitter.
Each chunk includes metadata: source filename and page number.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_pdfs(data_dir: str = "data") -> list:
    """
    Load all PDFs from the data directory and chunk them.
    
    Args:
        data_dir: Path to directory containing PDF files (default: "data")
    
    Returns:
        List of chunked documents with metadata (source filename, page number)
    
    Example:
        chunks = load_and_chunk_pdfs()
        print(f"Loaded {len(chunks)} chunks from PDFs")
    """
    
    # Ensure data directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} directory not found. Creating it...")
        os.makedirs(data_dir)
        return []
    
    # Get all PDF files
    pdf_files = list(Path(data_dir).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return []
    
    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")
    
    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,           # Characters per chunk
        chunk_overlap=50,         # Overlap between chunks
        separators=["\n\n", "\n", " ", ""]  # Split hierarchy
    )
    
    all_chunks = []
    
    # Load and chunk each PDF
    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")
        
        try:
            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            print(f"  Pages: {len(documents)}")
            
            # Chunk the documents
            chunks = splitter.split_documents(documents)
            
            # Ensure metadata has source and page info
            for chunk in chunks:
                # Add/override source filename (without path)
                chunk.metadata["source"] = pdf_path.name
                # page_number is already set by PyPDFLoader
                if "page" not in chunk.metadata and "page_number" not in chunk.metadata:
                    chunk.metadata["page"] = 0
                # Convert 0-based page to 1-based for display
                chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
                    
            
            all_chunks.extend(chunks)
            print(f"  Chunks: {len(chunks)}")
            
        except Exception as e:
            print(f"  Error loading {pdf_path.name}: {e}")
            continue
    
    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    return all_chunks


def clear_data_folder(data_dir: str = "data") -> None:
    """
    Delete all PDF files from the data directory.
    Used to clear old PDFs before processing new uploads.
    
    Args:
        data_dir: Path to directory to clear (default: "data")
    """
    if not os.path.exists(data_dir):
        return
    
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            pdf_file.unlink()
            print(f"Deleted: {pdf_file.name}")
        except Exception as e:
            print(f"Could not delete {pdf_file.name}: {e}")


def get_chunks_sample(chunks: list, num_samples: int = 3) -> None:
    """
    Print sample chunks for inspection.
    
    Args:
        chunks: List of document chunks
        num_samples: Number of samples to print
    """
    if not chunks:
        print("No chunks to display")
        return
    
    print(f"\n{'='*60}")
    print(f"Sample chunks (showing {min(num_samples, len(chunks))} of {len(chunks)})")
    print(f"{'='*60}")
    
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"\nChunk {i+1}")
        print(f"  Source: {chunk.metadata.get('source', 'Unknown')}")
        print(f"  Page: {chunk.metadata.get('page', 'Unknown')}")
        print(f"  Length: {len(chunk.page_content)} chars")
        print(f"  Content preview: {chunk.page_content[:100]}...")


if __name__ == "__main__":
    # Test the loader
    chunks = load_and_chunk_pdfs()
    if chunks:
        get_chunks_sample(chunks, num_samples=3)