import os
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

# ✅ Streamlit Cloud writable location
CHROMA_DIR = "/tmp/chroma"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Load embedding + LLM once
embedding = HuggingFaceEmbeddings()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# We'll store the active collection name here (set during processing)
_ACTIVE_COLLECTION = None


def _collection_name_from_path(pdf_path: str) -> str:
    """Create a deterministic collection name per document."""
    h = hashlib.md5(pdf_path.encode("utf-8")).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    safe = "".join(c if c.isalnum() else "_" for c in base)[:30]
    return f"{safe}_{h}"


def process_doc_to_chromadb(pdf_path: str):
    """
    pdf_path: full path to the uploaded PDF (e.g., /tmp/uploads/foo.pdf)
    """
    global _ACTIVE_COLLECTION

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    # ✅ Use lightweight loader (no unstructured/cv2)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    collection_name = _collection_name_from_path(pdf_path)

    # ✅ Persist to /tmp, not repo folder
    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )

    _ACTIVE_COLLECTION = collection_name
    return collection_name


def question_answer(question: str):
    global _ACTIVE_COLLECTION

    if not _ACTIVE_COLLECTION:
        raise RuntimeError("No document processed yet. Upload a PDF first.")

    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=CHROMA_DIR,
        collection_name=_ACTIVE_COLLECTION,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_with_sources = RunnableParallel(
        answer=rag_chain,
        sources=retriever
    )

    out = rag_with_sources.invoke(question)
    return out["answer"]
