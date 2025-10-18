import argparse
import json
import os
from rag_handler import RAGHandler
from llm_handler import LLMHandler

def build_rag_from_json(json_file):
    """
    Builds or rebuilds the RAG database from a JSON file of examples.

    Args:
        json_file (str): The path to the JSON file containing the RAG examples.
    """
    print("Initializing handlers to build RAG database...")
    try:
        # We only need the LLM handler for the abstraction part,
        # and the RAG handler for the storage part.
        llm_handler = LLMHandler()
        rag_handler = RAGHandler()
        print("Handlers initialized successfully.")
    except Exception as e:
        print(f"Error initializing handlers: {e}")
        print("Please ensure your config.json is present and Ollama is running.")
        return

    if not os.path.exists(json_file):
        print(f"Error: JSON file not found at '{json_file}'")
        return

    with open(json_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    print(f"Found {len(examples)} examples to add from '{json_file}'.\n")

    for i, example in enumerate(examples):
        print(f"--- Processing Example {i+1} ---")
        original_prompt = example.get("original_prompt", "")
        code = example.get("code", "")

        if not original_prompt or not code:
            print(f"Skipping example {i+1} due to missing 'original_prompt' or 'code'.")
            continue

        # Use the provided abstract prompt if it exists, otherwise generate it.
        if "abstract_prompt" in example:
            abstract_prompt = example["abstract_prompt"]
            print(f"Using provided abstract prompt: {abstract_prompt}")
        else:
            print(f"Generating abstract prompt with LLM...")
            abstract_prompt = llm_handler.abstract_user_prompt(original_prompt)

        print(f"Original Prompt: {original_prompt}")

        try:
            rag_handler.add_successful_automation(
                abstract_prompt=abstract_prompt,
                original_prompt=original_prompt,
                python_code=code
            )
        except Exception as e:
            print(f"Failed to add document to RAG index: {e}")

        print("-" * 20 + "\n")

    print("Finished building RAG database.")
    print("The vector_db directory should now be populated with the new examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a RAG database from a JSON file of examples.")
    parser.add_argument("json_file", help="The path to the JSON file containing the examples.")
    args = parser.parse_args()
    build_rag_from_json(args.json_file)