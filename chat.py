# from langchain_ollama import ChatOllama

# model = ChatOllama(
#     model="llama3.2",
#     temperature=0
# )

# response = model.invoke("What is cricket?")

# print(response.content)

# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model

# load_dotenv()

# model = init_chat_model(
#     "google_genai:gemini-3.6-flash",
#     max_retries=10,
#     timeout=120,
# )

# response = model.invoke("What is machine learning in one paragraph ?")

# print(response.content[0]["text"])

# import os

# from dotenv import load_dotenv
# from google import genai

# # Load .env file
# load_dotenv()

# # Create Gemini client
# client = genai.Client(
#     api_key=os.getenv("GOOGLE_API_KEY")
# )

# # Send prompt to Gemini
# response = client.models.generate_content(
#     model="gemini-3.6-flash",
#     contents="What is machine learning? Explain in simple words."
# )

# # Print response
# print(response.text)


# import os
# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )

# models = client.models.list()

# for model in models.data:
#     print(model.id)

# import os

# from dotenv import load_dotenv
# from groq import Groq

# load_dotenv()

# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY")
# )

# response = client.chat.completions.create(
#     model="openai/gpt-oss-20b",
#     messages=[
#         {
#             "role": "user",
#             "content": "What is machine learning? Explain in simple words."
#         }
#     ]
# )

# print(response.choices[0].message.content)

# from langchain_ollama import ChatOllama

# model = ChatOllama(
#     model="llama3.2:latest",
#     temperature=0
# )

# response = model.invoke(
#     "What is machine learning? Explain in simple words."
# )

# print(response.content)
from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7,
    max_tokens=20
)

response = model.invoke("What is Machine Learning?")

print(response.content)