import os
import requests
import chromadb
from chromadb.config import Settings

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

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


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def ingest_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)

        collection.add(
            ids=[f"{os.path.basename(filepath)}_{i}"],
            documents=[chunk],
            embeddings=[embedding]
        )

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_file("data.txt")
