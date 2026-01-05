from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


app = FastAPI(title="LangChain Gemini API")


#-----------------------------------
# Request Schema
#-----------------------------------
class QuesryRequest(BaseModel):
    query: str = Field(min_length=1)


#-----------------------------------
# Langchain Setup
#-----------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


OutPutParser = StrOutputParser()

chain = prompt | llm | OutPutParser



#-----------------------------------
# API Route
#-----------------------------------
@app.post("/ask")
async def ask_llm(payload: QuesryRequest):

    responce = chain.invoke({"question":payload.query})
    return{
        "question": payload.query,
        "answer": responce
    }
