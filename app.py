import streamlit as st
import requests
import chromadb
from chromadb.config import Settings

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "amalanai"
EMBED_MODEL = "nomic-embed-text"

# Connect Chroma
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    is_persistent=True
))
collection = client.get_or_create_collection("documents")


def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    return response.json()["embedding"]


def retrieve(query, k=3):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return "\n\n".join(results["documents"][0])


def generate_answer(context, question):
    prompt = f"""
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# Streamlit UI
st.title("🧠 Pure Ollama RAG")

query = st.text_input("Ask a question")

if query:
    with st.spinner("Thinking..."):
        context = retrieve(query)
        answer = generate_answer(context, query)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        st.write(context)