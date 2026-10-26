from langchain_community.document_loaders import WebBaseLoader

url="https://www.langchain.com/"

data = WebBaseLoader(url)
docs=data.load()
print (docs)