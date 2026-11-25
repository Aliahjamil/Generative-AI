from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=20,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of France?")

print(response.content)