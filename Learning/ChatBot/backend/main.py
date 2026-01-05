from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


from rag_engine import get_answer


app = FastAPI(title="LangChain Gemini API")


#-----------------------------------
# Request Schema
#-----------------------------------
class QuesryRequest(BaseModel):
    query: str = Field(min_length=1)


#-----------------------------------
# Langchain Setup
#-----------------------------------



# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0.7,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )


# OutPutParser = StrOutputParser()

# chain = prompt | llm | OutPutParser



#-----------------------------------
# API Route
#-----------------------------------
@app.post("/ask")
def ask_llm(payload: QuesryRequest):

    response = get_answer(payload.query)
    return {"answer": response}
