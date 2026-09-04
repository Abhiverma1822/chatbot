# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
# )
# model = ChatHuggingFace(llm=llm)
# response = model.invoke("who are you")

# print(response.content)
# from dotenv import load_dotenv
# import os

# load_dotenv()

# token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# print("Token loaded:", token is not None)

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# llm = HuggingFaceEndpoint(
#     repo_id="deepseek-ai/DeepSeek-R1",
#     huggingfacehub_api_token=token,
#     max_new_tokens=100,
#     temperature=0.7
# )

# model = ChatHuggingFace(llm=llm)

# response = model.invoke("Who are you?")

# print(response.content)

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    max_new_tokens=30,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Write a short poem ")

print(response.content)