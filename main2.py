from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
url="https://www.langchain.com/"

data = WebBaseLoader(url)
docs=data.load()
from langchain_mistralai import ChatMistralAI

load_dotenv()
model = ChatMistralAI(model ="ministral-8b-2512")

template = ChatPromptTemplate.from_messages(
     [("system","You are an ai that summarizes the text in 100 words"),
      ("human","{data}")
      ]
)

prompt=template.format_messages(data = docs)
 
result= model.invoke(prompt)

print(result.content)