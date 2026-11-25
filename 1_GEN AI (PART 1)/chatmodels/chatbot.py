from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

print("Starting...")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    max_tokens=300
)

print("Choose your AI mode")
print("Press 1 for Angry mode")
print("Press 2 for Funny mode")
print("Press 3 for Sad mode")

choice = int(input("Tell your response: "))

if choice == 1:
    mode = "You are an angry AI agent. Respond aggressively and impatiently."
elif choice == 2:
    mode = "You are a very funny AI agent. Respond with humor and jokes."
elif choice == 3:
    mode = "You are a sad AI agent. Respond in a sad and emotional way."
else:
    mode = "You are a helpful and professional AI agent."

messages = [
    SystemMessage(content=mode)
]

print("------------- Welcome! Type 0 to exit the application -------------")

while True:
    prompt = input("You : ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot :", response.content)

print("Goodbye!")

