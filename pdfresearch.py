import streamlit as st
import requests
import chromadb
import json
import uuid
from chromadb.config import Settings
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "amalanai"
EMBED_MODEL = "nomic-embed-text"

# --- Chroma ---
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    is_persistent=True
))
collection = client.get_or_create_collection("pdf_docs")


# --- Embeddings ---
def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


# --- Smart Chunking ---
def chunk_text(text, chunk_size=800):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --- PDF Ingestion ---
def ingest_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page_num, page in enumerate(reader.pages):
        text += page.extract_text() + "\n"

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[chunk],
            metadatas=[{
                "source": file.name,
                "chunk": i
            }],
            embeddings=[embedding]
        )


# --- Retrieval ---
def retrieve(query, k=4):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    context = ""
    sources = []

    for doc, meta in zip(documents, metadata):
        context += doc + "\n\n"
        sources.append(f"{meta['source']} (chunk {meta['chunk']})")

    return context, sources


# --- Streaming ---
def stream_generate(prompt):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
            except:
                continue


# --- UI ---
st.set_page_config(page_title="PDF Research Assistant", layout="wide")
st.title("📄 Local PDF Research Assistant")

# Upload
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.write(f"Ingesting: {file.name}")
        ingest_pdf(file)
    st.success("PDF(s) ingested successfully!")


# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about your PDFs..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    context, sources = retrieve(prompt)

    history_text = ""
    for msg in st.session_state.messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"""
You are a research assistant.

Answer strictly using the retrieved context.
If the answer is not in the documents, say:
"I could not find that in the uploaded PDFs."

Context:
{context}

Conversation:
{history_text}

Assistant:
"""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk in stream_generate(full_prompt):
            full_response += chunk
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    with st.expander("Sources"):
        for src in set(sources):
            st.write(src)