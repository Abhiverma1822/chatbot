from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# =========================
# 1. Embedding Model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# 2. ChromaDB
# =========================

vector_store = Chroma(
    collection_name="chat_long_term_memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# =========================
# 3. Add Memory
# =========================

def save_memory(text):
    vector_store.add_texts(
        texts=[text]
    )

    print("✅ Memory saved!")


# =========================
# 4. Search Memory
# =========================

def search_memory(query, k=3):

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


# =========================
# 5. Test
# =========================

if __name__ == "__main__":

    save_memory(
        "I am learning Python, Machine Learning and Generative AI."
    )

    save_memory(
        "I use VS Code for my programming projects."
    )

    save_memory(
        "I am building an AI chatbot using LangChain."
    )

    print("\n🔎 Searching memory...\n")

    results = search_memory(
        "What programming technologies am I learning?"
    )

    for result in results:
        print("Memory:", result.page_content)