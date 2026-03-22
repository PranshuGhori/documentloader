from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_xai import ChatXAI
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings()


docs = [
    Document(page_content="python is widely used for web development, data science, and machine learning",metadata={"source":"local"}),
    Document(page_content="pandas is a library used for data manipulation and analysis",metadata={"source":"assistant"}),
    Document(page_content="neural networks are a type of machine learning algorithm that are inspired by the structure of the human brain",metadata={"source":"user"}),
]

vectorstore = Chroma.from_documents(documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
    )



retriever = vectorstore.as_retriever()

docs = retriever.invoke("what is used for data manipulation and analysis")
print(docs[0].page_content)


# model = ChatXAI(
#     model="grok-4-fast-non-reasoning",
#     xai_api_key=os.getenv("XAI_API_KEY")
# )

# prompt = ChatPromptTemplate.from_messages([



