from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


file_path = '/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/ChatBot/SARCASM DETECTION IN PERSIAN.pdf'
loader = PyPDFLoader(file_path)
texts = loader.load()


full_text = " ".join([text.page_content for text in texts])



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
document = text_splitter.split_text(full_text)



docs = [
    Document(page_content=chunk, metadata={"source": "SARCASM DETECTION IN PERSIAN.pdf"})
    for chunk in document
]
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 

Context:
{context}

Question: {input}
""")


document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def get_answer(query: str):
    response = retrieval_chain.invoke({"input": query}) 
    return response["answer"]