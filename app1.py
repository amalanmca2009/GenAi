import streamlit as st
import requests
import chromadb
from chromadb.config import Settings
import json


OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "amalanai"
EMBED_MODEL = "nomic-embed-text"

# --- Connect Chroma ---
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    is_persistent=True
))
collection = client.get_or_create_collection("documents")


# --- Embedding ---
def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )
    return response.json()["embedding"]


# --- Retrieval ---
def retrieve(query, k=3):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    if results["documents"]:
        return "\n\n".join(results["documents"][0])
    return ""


# --- Streamed Generation ---
import json  # <-- add this at top


def stream_generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
            except json.JSONDecodeError:
                continue


# --- Streamlit UI ---
st.set_page_config(page_title="Local Chat Assistant", layout="wide")
st.title("🧠 Local RAG Chat Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask something..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    context = retrieve(prompt)

    # Build conversation history string
    history_text = ""
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"""
You are a helpful AI assistant.

Use the provided context when relevant.
If the answer is not in the context, respond naturally.

Context:
{context}

Conversation:
{history_text}

Assistant:
"""

    # Stream assistant reply
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in stream_generate(full_prompt):
            full_response += chunk
            message_placeholder.markdown(full_response)

    # Save assistant reply
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    with st.expander("Retrieved Context"):
        st.write(context)
