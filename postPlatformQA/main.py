import streamlit as st
import langchain_helper as lch
import textwrap

# video_url = "https://www.youtube.com/watch?v=lG7Uxts9SXs"
st.set_page_config(initial_sidebar_state="expanded")


st.title("Youtube assistant")
with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label="Post the youtube video URL",
            max_chars=50
        )
        question = st.sidebar.text_area(
            label="What's your question?",
            max_chars=50,
            key="query"
        )
        
        submit_button = st.form_submit_button(label="Submit")
        
if question and youtube_url:
    db = lch.youtube_to_vectorDB(youtube_url)
    response = lch.get_reponse_from_query(db, question)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))