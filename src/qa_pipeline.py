from pathlib import Path
import os
from dotenv import load_dotenv
import streamlit as st  # For Streamlit Cloud secrets fallback

# --- Chroma backend configuration BEFORE importing chroma/langchain_chroma ---
# Streamlit Cloud (and some slim containers) can have an older system sqlite (<3.35)
# which triggers chromadb's runtime check. Force the duckdb+parquet backend to avoid
# reliance on system sqlite. Also disable telemetry in ephemeral academic/demo usage.
os.environ.setdefault("CHROMA_DB_IMPL", "duckdb+parquet")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Attempt proactive sqlite replacement if available (some hosts ship old sqlite3)
try:
    import pysqlite3  # type: ignore
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules["pysqlite3"]
except Exception:
    pass  # Not fatal; fallback logic later will still handle errors

# Load environment variables from .env file
load_dotenv()
# Attempt resolution order: explicit env var -> Streamlit secrets -> default
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL") or st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")
VECTOR_BACKEND = (os.getenv("VECTOR_BACKEND") or st.secrets.get("VECTOR_BACKEND") or "faiss").lower()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Mitigate huggingface tokenizers fork warning unless user overrides
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Initialize Groq LLM client (handle missing key gracefully)
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    streaming=True
)

def load_and_chunk_file(filepath: Path):
    """Load a PDF file, chunk its content, and auto-generate metadata from filename."""
    loader = PyPDFLoader(str(filepath))
    docs = loader.load()

    # Auto-metadata parsing from filename
    filename = filepath.stem.lower()
    meta = {}
    if "grade" in filename:
        try:
            digits = [x for x in filename.split() if x.isdigit()]
            if digits:
                meta["grade"] = int(digits[0])
        except ValueError:
            meta["grade"] = None
    # Detect subject from filename keywords
    if "physics" in filename:
        meta["subject"] = "Physics"
    elif ("math" in filename) or ("mathematics" in filename):
        meta["subject"] = "Math"
    elif "biology" in filename:
        meta["subject"] = "Biology"
    elif "chemistry" in filename:
        meta["subject"] = "Chemistry"
    elif "english" in filename:
        meta["subject"] = "English"
    elif "history" in filename:
        meta["subject"] = "History" 
    elif "geography" in filename:
        meta["subject"] = "Geography"
    elif "civics" in filename:
        meta["subject"] = "Civics"
    elif "economics" in filename:
        meta["subject"] = "Economics"
    elif "social science" in filename:
        meta["subject"] = "Social Science"
    else:
        meta["subject"] = "General"

    # Split document into chunks for embedding and retrieval
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    # Attach metadata to each chunk
    for d in split_docs:
        d.metadata.update(meta)
        d.metadata["source_file"] = filepath.name
    return split_docs

def create_vectorstore(uploaded_files):
    """Build a vector store (FAISS by default, optional Chroma).

    Controlled by VECTOR_BACKEND env/secret: 'faiss' (default) or 'chroma'.
    """
    all_chunks = []
    for f in uploaded_files:
        try:
            chunks = load_and_chunk_file(f)
            all_chunks.extend(chunks)
        except Exception as e:
            st.warning(f"Failed to process {f.name}: {e}")
    if not all_chunks:
        raise ValueError("No valid documents to index.")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    if VECTOR_BACKEND == "chroma":
        try:
            # Modern client usage (avoids deprecated path)
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
            from langchain_chroma import Chroma  # type: ignore
            persist_dir = "chroma_db"
            client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
            collection_name = "documents"
            # Build from docs manually to ensure using existing or new collection
            vectorstore = Chroma.from_documents(
                all_chunks,
                embedding=embeddings,
                client=client,
                collection_name=collection_name,
            )
        except Exception as e:
            st.warning(f"Chroma failed ({e}); switching to FAISS in-memory.")
            from langchain_community.vectorstores import FAISS  # type: ignore
            vectorstore = FAISS.from_documents(all_chunks, embeddings)
    else:
        # Default FAISS simple path
        from langchain_community.vectorstores import FAISS  # type: ignore
        vectorstore = FAISS.from_documents(all_chunks, embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

def build_rag_chain(retriever):
    """Create a RAG chain for question answering using retrieved context."""
    template = """
    You are a helpful AI tutor. 
    Use ONLY the retrieved content from textbooks/notes to answer the student's question.
    Always provide a reference at the end like: "For more, see Grade {grade}, {subject}, file: {source_file}."

    Question: {question}
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_inputs(question):
        # Retrieve relevant docs
        docs = retriever.invoke(question)
        # Combine context from docs
        context = "\n".join([doc.page_content for doc in docs])
        # Extract metadata from the first doc (or customize as needed)
        meta = docs[0].metadata if docs else {}
        # Return formatted inputs for the prompt
        return {
            "context": context,
            "question": question,
            "grade": meta.get("grade", "N/A"),
            "subject": meta.get("subject", "N/A"),
            "source_file": meta.get("source_file", "N/A"),
        }

    # Build the chain: passthrough -> format inputs -> prompt -> LLM
    rag_chain = (
        RunnablePassthrough()  # Pass the question
        | format_inputs        # Format all required prompt variables
        | prompt
        | llm
    )
    return rag_chain

def build_summary_chain(file_path):
    """Create a chain to summarize the content of a textbook/note.

    Returns a runnable that accepts an (optional) topic or focus string
    and generates a summary grounded in the file's text.
    """
    chunks = load_and_chunk_file(file_path)
    context = "\n".join(doc.page_content for doc in chunks)
    template = """
    You are a helpful AI tutor.
    Provide a concise, student-friendly summary of the provided material.
    If the user supplies a focus/topic, emphasize that topic while still covering key ideas.

    Topic / Focus: {topic}
    Source Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def supply_inputs(user_topic: str = ""):
        return {"topic": user_topic or "(general overview)", "context": context}

    summary_chain = (RunnablePassthrough() | supply_inputs | prompt | llm)
    return summary_chain