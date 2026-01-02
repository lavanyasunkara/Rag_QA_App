import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#load environment variable
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

#load embedding model

embedding = HuggingFaceEmbeddings()

#load llm model
llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature =0)

def process_doc_to_chromadb(file_name):
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    texts = text_splitter.split_documents(documents)


    vector_store = Chroma.from_documents(
        documents = texts,
        embedding =embedding,
        persist_directory=f"{working_dir}/chroma",

    )
    return 0

def question_answer(question):
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory= f"{working_dir}/chroma",
    )
    retriever = vector_store.as_retriever()
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
    res= out["answer"]
    return res
