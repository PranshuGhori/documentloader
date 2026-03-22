import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

model = ChatXAI(
    model="grok-4-fast-non-reasoning",
    xai_api_key=os.getenv("XAI_API_KEY")
)

data = PyPDFLoader("document loaders/ps1.pdf").load()
docs = splitter.split_documents(data)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

splitter = splitter.split_documents(PyPDFLoader("document loaders/ps1.pdf").load())

chat = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that summarises the data comprehensively"), 
    ("user", "{data}")
])
prompt = chat.format_messages(data=docs[0].page_content)
result = model.invoke(prompt)
print(result.content)