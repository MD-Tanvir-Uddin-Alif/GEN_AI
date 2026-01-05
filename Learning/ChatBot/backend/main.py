from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv
import os


load_dotenv()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question:{question}")
    ]
)


st.title('Langchain Demo With GEMINI API')
input_text = st.text_input("Search the topic u want")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
    print(chain.invoke({'question': input_text}))