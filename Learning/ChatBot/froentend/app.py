import streamlit as st
import requests

st.title("LangChain Demo with Gemini (FastAPI)")

input_text = st.text_input("Search the topic you want")
submit = st.button("Ask")

if submit and input_text.strip():
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"query": input_text.strip()}
        )

        if response.status_code == 200:
            st.write(response.json()["answer"])
        else:
            st.error(response.text)

