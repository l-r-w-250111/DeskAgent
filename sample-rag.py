import os

import streamlit as st
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

INDEX_BASE_DIR: str = "./index"
INDEX_DIR = os.path.join(INDEX_BASE_DIR, "indexes")
DOCUMENTS_DIR = os.path.join(INDEX_BASE_DIR, "documents")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
INDEX_PATH = os.path.join(INDEX_DIR, "index.json")

if name == "__main__":
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "何か気になることはありますか？"}
        ]

    documents = SimpleDirectoryReader("data").load_data()
    os.makedirs(INDEX_BASE_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    llm = Ollama(
        model="gpt-oss:20b",
        request_timeout=360.0,
    )

    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    if os.path.exists(INDEX_PATH):
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_PATH)
        index = load_index_from_storage(storage_context, embed_model=embed_model, llm=llm)
    else:
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        index.storage_context.persist(INDEX_PATH)

    query_engine = index.as_query_engine(llm=llm)

    st.write("# Ollama チャット")
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    prompt: str | None = st.chat_input()
    if prompt is not None:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = query_engine.query(prompt + "\nこの質問に日本語で回答してください。")
        st.chat_message("assistant").write(response.response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.response}
        )
