import os
import streamlit as st

import rag_utility
from rag_utility import *

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Rag Question-Answer App")

upload_file = st.file_uploader("Upload a PDF file",type =["pdf"])

if upload_file is not None:
    save_path = os.path.join(working_dir,upload_file.name)
    with open(save_path,mode='wb') as f:
        f.write(upload_file.getbuffer())

    process_document = process_doc_to_chromadb(upload_file.name)
    st.info("Document processed successfully")

user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):
    answer = question_answer(user_question)
    st.markdown("LLM response:")
    st.markdown(answer)