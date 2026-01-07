from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_classic.tools.retriever import create_retriever_tool


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder





#---------------------------------------
# Reading from Arxiv(tool)
#---------------------------------------

arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=2000)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)




#---------------------------------------
# Reading from WikiPidia(tool)
#---------------------------------------

wiki_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=2000)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)




#---------------------------------------
# Reading from PDF
#---------------------------------------

file_path = '/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/ChatBot/SARCASM DETECTION IN PERSIAN.pdf'
loader = PyPDFLoader(file_path)
texts = loader.load()


full_text = " ".join([text.page_content for text in texts])



text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
document = text_splitter.split_text(full_text)



docs = [
    Document(page_content=chunk, metadata={"source": "SARCASM DETECTION IN PERSIAN.pdf"})
    for chunk in document
]

# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install sentence-transformers


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever()

retriever_tool = create_retriever_tool(retriever, 'SARCASM_DETECTION_IN_PERSIAN', "Search the Persian sarcasm detection research paper PDF for technical details, models, datasets, and experiments.")



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a research assistant. 
    Think step by step. Use tools when needed.
    """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# document_chain = create_stuff_documents_chain(llm, prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

tools = [wiki, arxiv, retriever_tool]


from langchain.agents import create_agent
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# parser = StrOutputParser()

def get_answer(query: str):
    response = agent_executor.invoke({"input": query})
    output = response["output"]

    if isinstance(output, list):
        final_text = []
        for part in output:
            if isinstance(part, dict) and "text" in part:
                final_text.append(part["text"])
            elif isinstance(part, str):
                final_text.append(part)
        return "".join(final_text)

    return str(output)
