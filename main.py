import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI

load_dotenv()

model = ChatXAI(
    model="grok-4-fast-non-reasoning",
    xai_api_key=os.getenv("XAI_API_KEY")
)

result = model.invoke("Hello, how are you?")

print(result.content)

