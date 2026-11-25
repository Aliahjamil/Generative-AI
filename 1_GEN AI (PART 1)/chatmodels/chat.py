
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=20
)

response = model.invoke("Write a short poem about Artificial Intelligence.")

print(response.content)
