# from langchain_community.document_loaders import WebBaseLoader
# import bs4


# loader = WebBaseLoader(web_path="https://docs.langchain.com/oss/python/integrations/document_loaders/web_base")


# text_documant = loader.load()
# print(text_documant)



from langchain_community.document_loaders import PyPDFLoader


file_path = '/home/md-tanvir-uddin-alif/Personal_Project/Gen_AI/Learning/ChatBot/SARCASM DETECTION IN PERSIAN.pdf'
loader = PyPDFLoader(file_path)
text = loader.load()
print(text)
