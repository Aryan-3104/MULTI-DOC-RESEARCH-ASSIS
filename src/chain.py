"""
RAG Chain Module

Builds the LLM chain using LangChain LCEL (Langchain Expression Language).
Uses Groq API with Llama 3.1 8B model.
Formats retrieved chunks and generates answers with source citations.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.documents import Document

from src.retriever import format_retrieved_chunks

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0  # Factual, deterministic answers
MAX_TOKENS = 1024


def get_groq_api_key() -> str:
    """
    Get Groq API key from environment.
    
    Raises:
        ValueError: If GROQ_API_KEY not set in .env
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in .env file. "
            "Please add: GROQ_API_KEY=your_key"
        )
    return api_key


def create_llm() -> ChatGroq:
    """
    Create a ChatGroq LLM instance.
    
    Uses llama-3.1-8b-instant model with temperature=0 for factual answers.
    
    Returns:
        ChatGroq LLM object
    """
    api_key = get_groq_api_key()
    
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
        timeout=30.0
    )
    
    print(f"✓ LLM initialized: {MODEL_NAME} (temp={TEMPERATURE})")
    return llm


def create_prompt() -> PromptTemplate:
    """
    Create the RAG prompt template.
    
    Instructions:
    - ALWAYS cite document and page number
    - Synthesize across ALL documents
    - Compare different perspectives when documents discuss same topic
    - Structure answer by document when comparing
    - Never claim information is unavailable if relevant chunks exist
    
    Returns:
        PromptTemplate object
    """
    template = """You are an expert research assistant analyzing multiple documents.

**Important Rules:**
1. ALWAYS cite which document and page your answer comes from
2. You MUST attempt to synthesize information across ALL provided documents
3. If different documents discuss the same topic differently, compare them explicitly
4. Never say you cannot find information if relevant chunks are provided
5. Structure your answer by document when comparing across papers
6. Format citations as: [Source: filename.pdf, Page X]

**Retrieved Documents:**
{context}

**User Question:**
{question}

**Answer:**"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    
    return prompt


def build_rag_chain(retriever, llm: ChatGroq = None) -> RunnablePassthrough:
    """
    Build the complete RAG chain using LangChain LCEL.
    
    Flow:
    1. User question → retriever fetches top-4 chunks
    2. Format chunks with metadata
    3. Pass to LLM with prompt
    4. LLM generates answer with citations
    
    Args:
        retriever: Document retriever from vectorstore
        llm: ChatGroq LLM (creates if None)
    
    Returns:
        LangChain LCEL chain
    
    Example:
        retriever = create_retriever(vectorstore)
        chain = build_rag_chain(retriever)
        response = chain.invoke({"question": "What is AI?"})
    """
    
    if llm is None:
        llm = create_llm()
    
    prompt = create_prompt()
    
    # LCEL Chain: Input → retrieve docs → format → prompt → LLM → output
    def retrieve_and_format(inputs):
        """Retrieve documents and format with metadata"""
        question = inputs["question"]
        docs = retriever.invoke(question)
        context = format_retrieved_chunks(docs)
        return {"context": context, "question": question}
    
    chain = (
        RunnablePassthrough()
        | retrieve_and_format
        | prompt
        | llm
    )
    
    print("✓ RAG chain built successfully")
    return chain


def invoke_chain(chain: RunnablePassthrough, question: str) -> str:
    """
    Invoke the RAG chain with a user question.
    
    Args:
        chain: RAG chain from build_rag_chain()
        question: User question
    
    Returns:
        Answer with citations
    """
    result = chain.invoke({"question": question})
    
    # Extract text if it's a message object
    if hasattr(result, "content"):
        return result.content
    return str(result)


if __name__ == "__main__":
    # Test chain creation (requires vectorstore and .env with GROQ_API_KEY)
    from src.embedder import load_embeddings, load_vectorstore
    from src.retriever import create_retriever
    
    try:
        embeddings = load_embeddings()
        vectorstore = load_vectorstore(embeddings)
        retriever = create_retriever(vectorstore, k=4)
        chain = build_rag_chain(retriever)
        
        # Test question
        test_question = "What is the main topic of the documents?"
        print(f"\nTest Question: {test_question}")
        print("-" * 60)
        answer = invoke_chain(chain, test_question)
        print(f"\nAnswer:\n{answer}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo test chain.py:")
        print("1. Set GROQ_API_KEY in .env")
        print("2. Ensure vectorstore exists (run loader.py + embedder.py first)")
