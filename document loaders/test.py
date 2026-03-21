import os
from langchain_community.document_loaders import TextLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
loader = TextLoader(os.path.join(current_dir, "notes.txt"))

documents = loader.load()
print(documents)