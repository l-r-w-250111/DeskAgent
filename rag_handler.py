import os
import json
import ollama
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from typing import List, Dict

class RAGHandler:
    """
    Handles the Retrieval-Augmented Generation functionality, including
    storing and retrieving successful automation scripts.
    """
    def __init__(self, config_path: str = "config.json", vector_db_path: str = "vector_db"):
        """
        Initializes the RAG handler, loading or creating the vector index.
        """
        self.vector_db_path = vector_db_path
        os.makedirs(self.vector_db_path, exist_ok=True)

        with open(config_path, 'r') as f:
            config = json.load(f)

        embedding_ollama_url = config.get('embedding_ollama_url', config['ollama_url'])

        # Pre-load and persist the embedding model
        try:
            print(f"Pre-loading embedding model from {embedding_ollama_url}...")
            client = ollama.Client(host=embedding_ollama_url)
            client.generate(model=config['embedding_model'], prompt=" ", options={'keep_alive': -1})
            print("Embedding model pre-loaded successfully.")
        except Exception as e:
            print(f"Could not pre-load embedding model: {e}")


        self.embed_model = OllamaEmbedding(
            model_name=config['embedding_model'],
            base_url=embedding_ollama_url,
        )

        try:
            print("Loading RAG index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=self.vector_db_path)
            self.index = load_index_from_storage(storage_context, embed_model=self.embed_model)
            print("RAG index loaded successfully.")
        except FileNotFoundError:
            print("No existing RAG index found. Creating a new one.")
            # We create an empty index. Documents will be added later.
            self.index = VectorStoreIndex.from_documents([], embed_model=self.embed_model)
            self.index.storage_context.persist(self.vector_db_path)
            print("New RAG index created.")

    def add_successful_automation(self, abstract_prompt: str, original_prompt: str, python_code: str):
        """
        Adds a successfully executed automation script to the vector index.
        The *abstracted* prompt is used as the content for retrieval.

        Args:
            abstract_prompt (str): The generalized version of the command.
            original_prompt (str): The initial user instruction.
            python_code (str): The Python code that successfully performed the task.
        """
        # We create a Document where the text is the ABSTRACT prompt for better searching,
        # and the original prompt and code are metadata.
        doc = Document(
            text=abstract_prompt,
            metadata={
                "python_code": python_code,
                "original_prompt": original_prompt
            }
        )

        try:
            print(f"Adding new document to RAG index for abstract prompt: {abstract_prompt}")
            self.index.insert(doc)
            self.index.storage_context.persist(self.vector_db_path)
            print("Document added and index persisted.")
        except Exception as e:
            print(f"Failed to add document to RAG index: {e}")

    def retrieve_similar_examples(self, user_prompt: str, top_k: int = 2) -> List[Dict[str, str]]:
        """
        Retrieves the most similar successful automation scripts from the index.

        Args:
            user_prompt (str): The new user instruction.
            top_k (int): The number of similar examples to retrieve.

        Returns:
            A list of dictionaries, each containing a 'prompt' and 'code'.
        """
        if len(self.index.docstore.docs) == 0:
            print("RAG index is empty. No examples to retrieve.")
            return []

        print(f"Querying RAG index for similar examples to: {user_prompt}")
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        retrieved_nodes = retriever.retrieve(user_prompt)

        examples = []
        if retrieved_nodes:
            for node in retrieved_nodes:
                # We search by the abstract prompt (node.get_content()),
                # but we use the *original* prompt for the few-shot example.
                original_prompt = node.metadata.get("original_prompt")
                code = node.metadata.get("python_code", "")
                if code and original_prompt:
                    examples.append({"prompt": original_prompt, "code": code})
            print(f"Retrieved {len(examples)} similar example(s).")
        else:
            print("No similar examples found.")

        return examples
