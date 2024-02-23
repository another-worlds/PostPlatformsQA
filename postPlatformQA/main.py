import streamlit as st
import langchain_helper as lch
import textwrap

st.set_page_config(initial_sidebar_state="expanded")


st.title("Postplatforms QA")
with st.sidebar:
    with st.form(key="my_form"):
        question = st.sidebar.text_area(
            label="What's your question?",
            max_chars=50,
            key="query"
        )
        
        submit_button = st.form_submit_button(label="Submit")
        
#mode = 'prod'
mode = 'test'

if mode == 'test':
    db = lch.docs_to_chroma()
    response = lch.db_to_agent_chain(db, question)
elif mode == 'prod':
    db = lch.docs_to_vectorDB()
    response = lch.db_to_retrieval_chain(db, question)
    
if question:
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))
