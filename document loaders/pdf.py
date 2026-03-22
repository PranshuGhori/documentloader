from langchain_community.document_loaders import PyPDFLoader



data = PyPDFLoader("document loaders/ps1.pdf")
documents = data.load()
print(len(documents))