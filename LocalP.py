import streamlit as st
import requests
import json

st.title("Local Ollama Chatbot (phi3)")

OLLAMA_URL = "http://localhost:11434/api/generate"

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your message..."):

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Stream assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "amalanai",
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        # Read streamed chunks
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                full_response += chunk.get("response", "")
                message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
