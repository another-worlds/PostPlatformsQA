import streamlit as st
import refactor as ref
import textwrap

st.set_page_config(initial_sidebar_state="expanded")
docs_path = "docs_cut.pdf"


st.title("Postplatforms QA")
with st.sidebar:
    with st.form(key="my_form"):
        question = st.sidebar.text_area(
            label="What's your question?",
            max_chars=50,
            key="query"
        )
        
        submit_button = st.form_submit_button(label="Submit")
        
if question:
    db = ref.docs_to_vectorDB(docs_path)
    response = ref.db_to_retrieval_chain(db, question)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))