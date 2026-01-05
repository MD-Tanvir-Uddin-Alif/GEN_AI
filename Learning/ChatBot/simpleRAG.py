from langchain_community.document_loaders import WebBaseLoader
import bs4


loader = WebBaseLoader(web_path="https://docs.langchain.com/oss/python/integrations/document_loaders/web_base")


text_documant = loader.load()
print(text_documant)



from langchain_community.document_loaders import PyPDFLoader


file_path = '/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/ChatBot/SARCASM DETECTION IN PERSIAN.pdf'
loader = PyPDFLoader(file_path)
texts = loader.load()
# print(text)

full_text = " ".join([text.page_content for text in texts])

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5)
document = text_splitter.split_text(full_text)
# print(document[1])


from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()
docs = [
    Document(page_content=chunk, metadata={"source": "SARCASM DETECTION IN PERSIAN.pdf"})
    for chunk in document
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)


query = "who are the authors"
results = db.similarity_search(query)
print(results[0])