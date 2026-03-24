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

chat = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant that summarises the data comprehensively"), 
    ("user", "{data}")
])
prompt = chat.format_messages(data=docs[0].page_content)
result = model.invoke(prompt)
print(result.content)