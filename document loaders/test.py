import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


current_dir = os.path.dirname(os.path.abspath(__file__))
loader = TextLoader(os.path.join(current_dir, "notes.txt"))

splitter = CharacterTextSplitter(separator="",
    chunk_size=10,
    chunk_overlap=1
)

documents = loader.load()

print(len(splitter.split_documents(documents)))

for i in splitter.split_documents(documents):
    print(i.page_content)
    print("\n")