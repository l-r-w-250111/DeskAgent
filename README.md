# Desktop Automation Assistant

This project is a powerful desktop automation tool that leverages local Large Language Models (LLMs) to translate natural language commands into executable Python scripts. It uses a Streamlit interface for user interaction and a RAG (Retrieval-Augmented Generation) system to learn from successful operations, continuously improving its performance.

## Features

- **Natural Language Control:** Automate desktop tasks by simply writing what you want to do.
- **LLM-Powered Code Generation:** Uses a local LLM (via Ollama) to generate Python code for mouse and keyboard actions.
- **Self-Improving:** A RAG system learns from successfully executed commands to improve the accuracy of future code generation.
- **OCR Integration:** Utilizes EasyOCR to read text on the screen, enabling the agent to understand the current state of the UI.
- **User-Friendly Interface:** A simple web-based UI built with Streamlit for easy operation and monitoring.
- **Configurable:** Easily configure the Ollama server URL and the specific models used for operation, evaluation, and embeddings.
- **Retry Mechanism:** Automatically retries a failed operation a configurable number of times.

## How It Works

The automation process follows these steps:
1.  **User Command:** The user enters a command in the Streamlit UI (e.g., "Open Notepad and type 'Hello, World!'").
2.  **RAG-Enhanced Prompting:** The system first generalizes the user's command and searches its knowledge base for similar successful past operations. These examples are then added to the prompt to improve the LLM's accuracy.
3.  **Code Generation:** The "Operation LLM" receives the user's command, a screenshot of the current screen (with OCR results), the screen resolution, and any relevant RAG examples. It then generates a Python script using the `pyautogui` library to perform the requested actions.
4.  **Execution:** The generated Python script is executed by the `DesktopController`.
5.  **LLM Evaluation:** After execution, an "Evaluation LLM" compares the screen's state before and after the operation to determine if the task was completed successfully.
6.  **User Confirmation & Learning:** If the LLM evaluation is successful, the user is asked for final confirmation. If the user confirms, the generalized command and the successful script are vectorized and saved to the RAG knowledge base for future use.
7.  **Retry on Failure:** If either the LLM or the user reports a failure, the entire process is repeated up to the configured maximum number of retries.

## Requirements

Before you begin, ensure you have the following installed:

- **Ollama:** The application relies on a running Ollama instance to serve the local LLMs. You can download it from [https://ollama.com/](https://ollama.com/).
- **Python 3.8+**

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `torch` and `torchvision` can be large. If you have a CUDA-enabled GPU, you may want to install a GPU-compatible version for better performance.*

3.  **Pull the required Ollama models:**
    This application uses several models by default. You can pull them using the following commands:
    ```bash
    ollama pull gemma3:12b  # For operation and evaluation
    ollama pull embeddinggemma   # For RAG embeddings
    ```

4.  **Verify Ollama is running:**
    Make sure the Ollama application is running on your machine.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Automate:**
    - Use the "Automation" page to enter your commands.
    - Monitor the progress in the "Logs" section.
    - Confirm or deny the success of an operation when prompted.

## Configuration

The application's settings can be configured in two ways:

1.  **`config.json` file:**
    You can create a `config.json` file in the root directory to override the default settings.
    ```json
    {
        "ollama_url": "http://localhost:11434",
        "operation_model": "gemma3:12b",
        "evaluation_model": "gemma3:12b",
        "embedding_model": "embeddinggemma",
        "max_retries": 3
    }
    ```

2.  **Settings UI:**
    Alternatively, you can navigate to the "Settings" page in the Streamlit application to change and save these values through the UI.

## Populating the Knowledge Base

You can pre-populate the RAG knowledge base with your own examples from a JSON file. This is useful for providing the system with a set of known successful operations before it starts learning from user interactions.

The `rag_examples.json` file provides a template for how to structure the data. To add the examples from this file to the knowledge base, run the following command:

```bash
python build_rag_from_json.py rag_examples.json
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

For information on the licenses of the third-party libraries used in this project, please see the [ThirdPartyLicenses.md](ThirdPartyLicenses.md) file.