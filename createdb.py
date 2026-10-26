from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
load_dotenv()

data = PyPDFLoader("document_loaders/egypt.pdf")
docs= data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap =200
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding= embedding_model,
    persist_directory="chroma_db"
)
# model = ChatMistralAI(model ="ministral-8b-2512")

# template = ChatPromptTemplate.from_messages(
#      [("system","You are an ai that summarizes the text in 100 words"),
#       ("human","{data}")
#       ]
# )

# prompt=template.format_messages(data = docs[0].page_content)

# result= model.invoke(prompt)

# print(result.content)