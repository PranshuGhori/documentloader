from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document loaders/ps1.pdf")
documents = data.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunk = splitter.split_documents(documents)
print(chunk[0].page_content)
