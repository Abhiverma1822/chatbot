from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    max_new_tokens=200,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)


print("🤖 Hugging Face Chatbot")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = model.invoke(user_input)

    print("Bot:", response.content)