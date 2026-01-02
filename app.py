import os
from pathlib import Path
import streamlit as st
from rag_utility import process_doc_to_chromadb, question_answer

st.title("Rag Question-Answer App")

UPLOAD_DIR = Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

upload_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if upload_file is not None:
    save_path = UPLOAD_DIR / upload_file.name
    with open(save_path, "wb") as f:
        f.write(upload_file.getbuffer())

    process_doc_to_chromadb(str(save_path))   # ✅ pass full path
    st.success("Document processed successfully ✅")
    st.session_state["doc_ready"] = True

user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    if not st.session_state.get("doc_ready"):
        st.error("Upload a document first.")
    else:
        answer = question_answer(user_question)
        st.markdown("### LLM response:")
        st.markdown(answer)
