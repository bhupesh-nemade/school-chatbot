from langchain_openai import ChatOpenAI
from config import OPENROUTER_API_KEY

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

response = llm.invoke("Say OK")
print(response.content)