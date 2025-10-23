import streamlit as st
import json
import os
import time
from PIL import Image
import numpy as np
import logging

# Import the handlers
from llm_singleton import get_llm_handler
from rag_handler import RAGHandler
# from ocr_helper import get_all_ocr_results, draw_ocr_results # Moved to run_automation_flow

import subprocess

# --- Handler Initialization ---
@st.cache_resource
def get_desktop_controller():
    """Initializes and returns a cached DesktopController instance."""
    from desktop_controller import DesktopController
    return DesktopController()


# --- Playwright Setup ---
@st.cache_resource
def setup_playwright():
    """
    Ensures Playwright browsers are installed. This is cached so it only runs once.
    """
    append_log("[SETUP] Checking Playwright installation...")
    try:
        # Check if browsers are installed by trying to access the command
        subprocess.check_output(['playwright', 'install', '--with-deps'], stderr=subprocess.STDOUT, text=True)
        append_log("[SUCCESS] Playwright browsers are installed.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        append_log("[INFO] Playwright browsers not found or error during check. Attempting installation...")
        try:
            install_process = subprocess.run(
                ['playwright', 'install', '--with-deps'],
                capture_output=True, text=True, check=True
            )
            append_log("[SUCCESS] Playwright browsers installed successfully.")
            append_log(install_process.stdout)
        except Exception as install_error:
            append_log(f"[ERROR] Failed to install Playwright browsers: {install_error}")
            st.error("Failed to install Playwright components. Please run 'playwright install --with-deps' manually in your terminal.")
            st.stop()


# --- Configuration and Logging ---
CONFIG_FILE = "config.json"
LOG_FILE = "app.log"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, mode='w'),
                              logging.StreamHandler()])

def load_config():
    """Loads configuration from JSON file."""
    if not os.path.exists(CONFIG_FILE):
        st.error(f"Configuration file '{CONFIG_FILE}' not found. Please create it.")
        st.stop()
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config):
    """Saves configuration to JSON file."""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

def append_log(message):
    """Appends a message to the log display in the Streamlit UI."""
    logging.info(message)
    st.session_state.logs += f"[{time.strftime('%H:%M:%S')}] {message}\n"


def cleanup_temp_files(files: list):
    """Deletes temporary files like screenshots."""
    append_log(f"Cleaning up {len(files)} temporary file(s)...")
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                append_log(f"Deleted {file_path}")
        except Exception as e:
            append_log(f"Error deleting {file_path}: {e}")

# --- Main Application Logic ---
def run_automation_flow(command: str, cdp_url: str = ""):
    """Orchestrates the entire automation process from command to execution."""
    append_log(f"Received command: '{command}'")
    if cdp_url:
        append_log(f"Using browser CDP endpoint: {cdp_url}")
    st.session_state.screenshots_to_cleanup = []

    try:
        # 1. Initialization
        append_log("Initializing handlers...")
        config = load_config()
        controller = get_desktop_controller()
        llm_handler = get_llm_handler()
        rag_handler = RAGHandler()
        max_retries = config.get('max_retries', 3)
        operation_successful = False
        final_code = ""

        # 2. Get Abstract Prompt for RAG
        append_log("Generalizing user command for RAG search...")
        abstract_prompt = llm_handler.abstract_user_prompt(command)
        st.session_state.abstract_prompt_for_saving = abstract_prompt
        append_log(f"Generalized command: '{abstract_prompt}'")

        for attempt in range(max_retries):
            append_log(f"--- Attempt {attempt + 1} of {max_retries} ---")

            # 3. Capture "Before" State
            from ocr_helper import get_all_ocr_results, draw_ocr_results # Import here
            append_log("Capturing 'before' screen and performing OCR...")
            before_screenshot_img, ocr_results = get_all_ocr_results()
            before_screenshot_path = controller.save_screenshot(before_screenshot_img, "before")
            st.session_state.screenshots_to_cleanup.append(before_screenshot_path)

            # Visualize OCR results
            visualized_img = draw_ocr_results(before_screenshot_img.copy(), ocr_results)
            st.image(visualized_img, caption=f"OCR Visualization (Attempt {attempt+1})")

            # For RAG, we only need the text content
            ocr_texts_for_rag = [{'text': text} for _, text, _ in ocr_results]

            # 4. RAG Search
            append_log("Searching for similar successful examples (RAG)...")
            rag_examples = rag_handler.retrieve_similar_examples(abstract_prompt)
            if rag_examples:
                append_log(f"[INFO] Found {len(rag_examples)} relevant example(s).")
            else:
                append_log("[INFO] No similar examples found. Proceeding with base model.")

            # 5. Generate Code
            screen_size = controller.get_screen_size()
            append_log(f"Generating automation code (Screen: {screen_size[0]}x{screen_size[1]})...")
            generated_code = llm_handler.generate_automation_code(
                command,
                screen_size,
                before_screenshot_path,
                rag_examples,
                cdp_url=cdp_url
            )

            if not generated_code:
                append_log("[ERROR] LLM failed to generate code. Retrying...")
                time.sleep(2)
                continue

            append_log("[SUCCESS] Code Generated:")
            st.code(generated_code, language='python')
            final_code = generated_code

            # 6. Execute Code
            append_log("Executing generated code...")
            try:
                controller.execute_code(generated_code)
                st.session_state.screenshots_to_cleanup.extend(["temp_automation_script.py", "script_error.log", "script_output.log"])
                append_log("[SUCCESS] Code execution command sent. Waiting for results...")
                time.sleep(3)  # Wait for the script to execute and write logs

                # Crash Detection
                if os.path.exists("script_error.log"):
                    with open("script_error.log", "r", encoding='utf-8') as f:
                        error_output = f.read().strip()
                    if error_output:
                        append_log(f"[ERROR] Script crashed. Error:\n{error_output}")
                        st.code(error_output, language='log')
                        append_log("Retrying...")
                        time.sleep(2)
                        continue # Move to the next attempt

            except Exception as e:
                append_log(f"[ERROR] Code execution failed to launch: {e}. Retrying...")
                time.sleep(2)
                continue

            # 7. Evaluate Result
            append_log("Capturing 'after' screen for evaluation...")
            after_screenshot_img = controller.capture_screenshot()
            after_screenshot_path = controller.save_screenshot(after_screenshot_img, "after")
            st.session_state.screenshots_to_cleanup.append(after_screenshot_path)
            st.image(after_screenshot_img, caption=f"Screen After Attempt {attempt+1}")

            # --- OCR-based evaluation for typing tasks ---
            if any(keyword in command.lower() for keyword in ["type", "enter", "入力"]):
                append_log("Performing OCR-based validation for typing task...")
                # Extract the text to be typed from the generated code
                # This is a bit brittle, assumes pyperclip.copy("text")
                try:
                    import ast

                    # Safely parse the code to find the text to be typed
                    text_to_find = None
                    tree = ast.parse(final_code)
                    for node in ast.walk(tree):
                        # Check for pyperclip.copy('...')
                        if isinstance(node, ast.Call) and \
                           isinstance(node.func, ast.Attribute) and \
                           node.func.attr == 'copy' and \
                           isinstance(node.func.value, ast.Name) and \
                           node.func.value.id == 'pyperclip' and \
                           node.args and isinstance(node.args[0], ast.Str):
                            text_to_find = node.args[0].s
                            break
                        # Check for pyautogui.typewrite('...')
                        if isinstance(node, ast.Call) and \
                           isinstance(node.func, ast.Attribute) and \
                           node.func.attr == 'typewrite' and \
                           isinstance(node.func.value, ast.Name) and \
                           node.func.value.id == 'pyautogui' and \
                           node.args and isinstance(node.args[0], ast.Str):
                            text_to_find = node.args[0].s
                            break

                    if not text_to_find:
                        raise ValueError("Could not find text being typed in the generated code.")

                    append_log(f"Searching for text '{text_to_find}' in the 'after' screenshot...")

                    # Get OCR results from the 'after' screenshot
                    _, after_ocr_results = get_all_ocr_results(after_screenshot_path)

                    found_text = False
                    for _, text, _ in after_ocr_results:
                        if text_to_find in text:
                            found_text = True
                            break

                    if found_text:
                        append_log(f"[SUCCESS] OCR validation: Found '{text_to_find}'.")
                        operation_successful = True
                        break # Exit retry loop
                    else:
                        append_log(f"[ERROR] OCR validation: Did not find '{text_to_find}'. Retrying...")

                except IndexError:
                    append_log("[WARNING] Could not parse text from code for OCR validation. Falling back to LLM.")
                    # Fallback to LLM evaluation if parsing fails
                    is_success = llm_handler.evaluate_operation(command, final_code, before_screenshot_path, after_screenshot_path)
                    if is_success:
                        operation_successful = True
                        break

            else:
                # --- LLM-based evaluation for other tasks ---
                append_log("Asking LLM to evaluate the result...")
                is_success = llm_handler.evaluate_operation(command, final_code, before_screenshot_path, after_screenshot_path)

                if is_success:
                    append_log("[SUCCESS] LLM evaluation: SUCCESS.")
                    operation_successful = True
                    break  # Exit retry loop
                else:
                    append_log("[ERROR] LLM evaluation: FAILURE. Retrying...")

        # 8. Final User Confirmation
        if operation_successful:
            st.session_state.validation_pending = {"prompt": command, "code": final_code}
        else:
            append_log("[ERROR] Automation failed after all retries.")
            st.error("The automation could not be completed successfully.")
            cleanup_temp_files(st.session_state.screenshots_to_cleanup)

    except Exception as e:
        error_message = f"A critical error occurred in the automation flow: {e}"
        append_log(f"[ERROR] {error_message}")
        st.error(error_message)
        st.warning("Please ensure Ollama is running and all required models are available.")
        cleanup_temp_files(st.session_state.get("screenshots_to_cleanup", []))

# --- Streamlit UI ---
def main_page():
    st.title("Desktop Automation Assistant 🤖")
    st.write("Enter a command to automate a desktop task. The assistant will write and execute the code for you.")

    # Initialize session state
    if 'validation_pending' not in st.session_state:
        st.session_state.validation_pending = None
    if 'abstract_prompt_for_saving' not in st.session_state:
        st.session_state.abstract_prompt_for_saving = ""

    user_command = st.text_area("Your Command:", height=100, key="user_command_input")

    if st.button("▶️ Run Automation", use_container_width=True):
        if user_command:
            # Reset state for a new run
            st.session_state.logs = ""
            st.session_state.validation_pending = None
            with st.spinner("Automation in progress... Please wait."):
                config = load_config()
                cdp_url = config.get("cdp_url", "")
                run_automation_flow(user_command, cdp_url=cdp_url.strip())
        else:
            st.warning("Please enter a command.")

    # --- Validation Section ---
    if st.session_state.get("validation_pending"):
        st.success("LLM reported success! Please provide your final confirmation.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Success", use_container_width=True):
                append_log("[SUCCESS] User confirmed SUCCESS.")
                pending_data = st.session_state.validation_pending
                try:
                    rag_handler = RAGHandler()
                    rag_handler.add_successful_automation(
                        st.session_state.abstract_prompt_for_saving,
                        pending_data['prompt'],
                        pending_data['code']
                    )
                    append_log("[SUCCESS] Saved operation to knowledge base for future use.")
                    st.success("Great! This successful operation will improve future automations.")
                except Exception as e:
                    append_log(f"[ERROR] Failed to save to knowledge base: {e}")
                    st.error(f"Could not save to RAG database: {e}")

                cleanup_temp_files(st.session_state.get("screenshots_to_cleanup", []))
                st.session_state.validation_pending = None

        with col2:
            if st.button("Report Failure", use_container_width=True):
                append_log("[INFO] User reported FAILURE. This example will not be saved.")
                st.warning("Thank you for the feedback. This operation will be discarded.")
                cleanup_temp_files(st.session_state.get("screenshots_to_cleanup", []))
                st.session_state.validation_pending = None

    st.code(st.session_state.logs, language="log")


def settings_page():
    st.title("⚙️ Settings")
    st.write("Configure the settings for the Ollama server and models.")

    config = load_config()

    with st.form("settings_form"):
        ollama_url = st.text_input("Ollama URL (for main LLMs)", value=config.get('ollama_url'))
        embedding_ollama_url = st.text_input("Ollama URL (for embedding model)", value=config.get('embedding_ollama_url'))
        operation_model = st.text_input("Operation Model", value=config.get('operation_model'))
        evaluation_model = st.text_input("Evaluation Model", value=config.get('evaluation_model'))
        embedding_model = st.text_input("Embedding Model", value=config.get('embedding_model'))
        max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=config.get('max_retries', 3))
        cdp_url = st.text_input(
            "Browser CDP Endpoint URL (Optional)",
            value=config.get('cdp_url', ''),
            placeholder="http://localhost:9222",
            help="To automate an existing browser, start it with a debugging port (e.g., `chrome.exe --remote-debugging-port=9222`) and enter the CDP URL here."
        )

        submitted = st.form_submit_button("Save Settings")
        if submitted:
            new_config = {
                "ollama_url": ollama_url,
                "embedding_ollama_url": embedding_ollama_url,
                "operation_model": operation_model,
                "evaluation_model": evaluation_model,
                "embedding_model": embedding_model,
                "max_retries": max_retries,
                "cdp_url": cdp_url
            }
            save_config(new_config)
            st.success("Settings saved successfully!")

# --- App Entrypoint ---
if __name__ == "__main__":
    st.set_page_config(page_title="Desktop Automation Assistant", layout="wide")

    # Initialize session state first
    if 'logs' not in st.session_state:
        st.session_state.logs = "Logs will appear here...\n"

    # Run setup
    setup_playwright()

    st.sidebar.title("Navigation")
    page_options = ["Automation", "Settings"]
    page = st.sidebar.radio("Go to", page_options)

    if page == "Automation":
        main_page()
    elif page == "Settings":
        settings_page()
