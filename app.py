import streamlit as st
import json
import os
import time
from PIL import Image
import numpy as np
import logging

# Import the handlers
from llm_handler import LLMHandler
from desktop_controller import DesktopController
from rag_handler import RAGHandler
from ocr_helper import get_all_ocr_results, draw_ocr_results

import subprocess

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
def run_automation_flow(command: str):
    """Orchestrates the entire automation process from command to execution."""
    append_log(f"Received command: '{command}'")
    st.session_state.screenshots_to_cleanup = []

    try:
        # 1. Initialization
        append_log("Initializing handlers...")
        config = load_config()
        controller = DesktopController()
        llm_handler = LLMHandler()
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
            append_log("Capturing 'before' screen and performing OCR...")
            before_screenshot_img, ocr_results = get_all_ocr_results()
            before_screenshot_path = controller.save_screenshot(before_screenshot_img, "before")
            st.session_state.screenshots_to_cleanup.append(before_screenshot_path)

            # Visualize OCR results
            visualized_img = draw_ocr_results(before_screenshot_img.copy(), ocr_results)
            st.image(visualized_img, caption=f"OCR Visualization (Attempt {attempt+1})")

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
            generated_code = llm_handler.generate_automation_code(command, screen_size, before_screenshot_path, rag_examples)

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
                # Add the temp script to the cleanup list
                st.session_state.screenshots_to_cleanup.append("temp_automation_script.py")
                append_log("[SUCCESS] Code execution command sent.")
                time.sleep(2)  # Wait for UI to settle
            except Exception as e:
                append_log(f"[ERROR] Code execution failed: {e}. Retrying...")
                time.sleep(2)
                continue

            # 7. Evaluate Result
            append_log("Capturing 'after' screen for evaluation...")
            after_screenshot_img = controller.capture_screenshot()
            after_screenshot_path = controller.save_screenshot(after_screenshot_img, "after")
            st.session_state.screenshots_to_cleanup.append(after_screenshot_path)
            st.image(after_screenshot_img, caption=f"Screen After Attempt {attempt+1}")

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
                run_automation_flow(user_command)
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
                st.rerun()

        with col2:
            if st.button("Report Failure", use_container_width=True):
                append_log("[INFO] User reported FAILURE. This example will not be saved.")
                st.warning("Thank you for the feedback. This operation will be discarded.")
                cleanup_temp_files(st.session_state.get("screenshots_to_cleanup", []))
                st.session_state.validation_pending = None
                st.rerun()

    st.code(st.session_state.logs, language="log")


def settings_page():
    st.title("⚙️ Settings")
    st.write("Configure the settings for the Ollama server and models.")

    config = load_config()

    with st.form("settings_form"):
        ollama_url = st.text_input("Ollama URL", value=config.get('ollama_url'))
        operation_model = st.text_input("Operation Model", value=config.get('operation_model'))
        evaluation_model = st.text_input("Evaluation Model", value=config.get('evaluation_model'))
        embedding_model = st.text_input("Embedding Model", value=config.get('embedding_model'))
        max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=config.get('max_retries', 3))

        submitted = st.form_submit_button("Save Settings")
        if submitted:
            new_config = {
                "ollama_url": ollama_url,
                "operation_model": operation_model,
                "evaluation_model": evaluation_model,
                "embedding_model": embedding_model,
                "max_retries": max_retries
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