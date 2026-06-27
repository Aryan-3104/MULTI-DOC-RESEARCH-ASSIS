"""
RAG Chain Module

Builds the LLM chain using LangChain LCEL (Langchain Expression Language).
Uses Google Gemini 2.5 Flash via ChatGoogleGenerativeAI.
Formats retrieved chunks and generates answers with source citations.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.documents import Document

from src.retriever import format_retrieved_chunks

# Load environment variables
load_dotenv()

# Model configuration
MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0  # Factual, deterministic answers
MAX_TOKENS = 1024


def get_google_api_key() -> str:
    """
    Get Google API key from environment.
    
    Raises:
        ValueError: If GOOGLE_API_KEY not set in .env
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in .env file. "
            "Please add: GOOGLE_API_KEY=your_key"
        )
    return api_key


def create_llm() -> ChatGoogleGenerativeAI:
    """
    Create a ChatGoogleGenerativeAI LLM instance.
    
    Uses gemini-2.5-flash model with temperature=0 for factual answers.
    
    Returns:
        ChatGoogleGenerativeAI LLM object
    """
    api_key = get_google_api_key()
    
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        google_api_key=api_key,
    )
    
    print(f"✓ LLM initialized: {MODEL_NAME} (temp={TEMPERATURE})")
    return llm


def create_prompt() -> PromptTemplate:
    """
    Create the RAG prompt template.
    
    Instructions:
    - Cite which document the answer came from
    - Synthesize across multiple documents when needed
    - Say "I could not find this" if context insufficient
    
    Returns:
        PromptTemplate object
    """
    template = """You are a helpful research assistant. Use the provided document chunks to answer the user's question.

**Important Rules:**
1. ALWAYS cite which document and page your answer comes from
2. If the question requires info from multiple documents, synthesize them clearly
3. If the documents don't contain enough info to answer, say: "I could not find sufficient information to answer this question in the provided documents."
4. Be concise but comprehensive
5. Format citations as: [Source: filename.pdf, Page X]

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


def build_rag_chain(retriever, llm: ChatGoogleGenerativeAI = None) -> RunnablePassthrough:
    """
    Build the complete RAG chain using LangChain LCEL.
    
    Flow:
    1. User question → retriever fetches top-4 chunks
    2. Format chunks with metadata
    3. Pass to LLM with prompt
    4. LLM generates answer with citations
    
    Args:
        retriever: Document retriever from vectorstore
        llm: ChatGoogleGenerativeAI LLM (creates if None)
    
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
    # Test chain creation (requires vectorstore and .env with GOOGLE_API_KEY)
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
        print("1. Set GOOGLE_API_KEY in .env")
        print("2. Ensure vectorstore exists (run loader.py + embedder.py first)")
