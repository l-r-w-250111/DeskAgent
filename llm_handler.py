import json
import ollama
from typing import Dict, Any, List

class LLMHandler:
    """
    Handles all interactions with the Ollama server, including code generation,
    evaluation, and embedding.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initializes the Ollama clients based on the provided configuration.
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.ollama_url = self.config['ollama_url']
        self.operation_model = self.config['operation_model']
        self.evaluation_model = self.config['evaluation_model']
        self.embedding_model = self.config['embedding_model']

        self.client = ollama.Client(host=self.ollama_url)
        print("LLMHandler initialized. Connected to Ollama.")

    def generate_automation_code(self, user_prompt: str, screen_size: tuple[int, int], screenshot_path: str, rag_examples: List[Dict[str, str]] = []) -> str:
        """
        Generates Python code for desktop automation.

        Args:
            user_prompt (str): The user's instruction.
            screen_size (tuple[int, int]): A tuple (width, height) of the screen.
            screenshot_path (str): The path to the current screenshot.
            rag_examples (List[Dict[str, str]]): A list of successful prompt/code examples from RAG.

        Returns:
            str: The generated Python code, ready for execution.
        """
        width, height = screen_size
        system_prompt = f"""
You are an expert Python programmer specializing in desktop automation. Your task is to write a Python script to accomplish the user's goal.

**DECISION FRAMEWORK: Choose the Right Tool**

First, analyze the user's prompt to determine the type of operation.

1.  **Browser-based Operation:**
    -   If the prompt contains keywords like "browser", "website", "URL", "Google", "search for", "go to", or a web address, you **MUST** use the `playwright` **Async API**.
    -   **Playwright Async Workflow:**
        -   **CRITICAL ASYNC HANDLING:** To prevent event loop errors, you **MUST** use the `nest_asyncio` library. The script must start with `import nest_asyncio` and `nest_asyncio.apply()`. This allows `asyncio.run()` to work correctly in all environments.
        -   **CRITICAL IMPORTS:** Your script **MUST** import `asyncio`, `nest_asyncio`, and `async_playwright`.
        -   **Main Function:** All playwright logic **MUST** be wrapped in an `async def main():` function.
        -   **Execution:** The script **MUST** end with `asyncio.run(main())`. Do **NOT** use `if __name__ == "__main__":`.
        -   **API Calls:** All Playwright API calls **MUST** be preceded by `await`.
        -   **Example Structure:**
            ```python
            import asyncio
            import nest_asyncio
            from playwright.async_api import async_playwright

            nest_asyncio.apply()

            async def main():
                async with async_playwright() as p:
                    browser = await p.chromium.launch(headless=False)
                    page = await browser.new_page()
                    await page.goto("https://www.google.com")
                    # ... more actions using await ...
                    await page.wait_for_load_state('networkidle')
                    # CRITICAL: Pause at the end to keep the browser open.
                    await page.pause()

            asyncio.run(main())
            ```
        -   **Locators:** Use robust locators like `page.locator('textarea[name=\"q\"]')`.
        -   **STRICT RULE FOR SEARCHING:** To perform a search, you **MUST** first fill the search input field, and then call `await page.locator(...).press('Enter')` on that **SAME** locator. **DO NOT** attempt to find and click a separate "Search" button. This is unreliable.

2.  **Desktop/Application Operation (Non-Browser):**
    -   If the prompt refers to a desktop application like "Notepad", "Calculator", or general GUI interactions, you **MUST** use `pyautogui`, `pygetwindow`, and `ocr_helper`.
    -   **PyAutoGUI Workflow:**
        1.  Launch the app with `subprocess.Popen()`.
        2.  Wait with `time.sleep()`.
        3.  Find and activate the window with `pygetwindow`.
        4.  Use `ocr_helper.find_text_coordinates()` to find buttons or text fields.
        5.  Use `pyautogui` to click and type.

**CRITICAL RULE FOR TYPING TEXT (for both tools):**
-   You MUST extract the literal text from the user's command and embed it directly in the function call.
-   **PyAutoGUI (non-ASCII):** `pyperclip.copy("こんにちは")` then `pyautogui.hotkey('ctrl', 'v')`.
-   **Playwright:** `page.get_by_role(...).fill("こんにちは")`. Playwright handles non-ASCII text automatically.
-   **DO NOT** use intermediate variables like `text_to_type`.

**STRICT OUTPUT RULES:**
-   Provide ONLY the Python code block. No explanations or comments.
-   Import all necessary libraries at the top of the script.
-   Screen Resolution: `{width}x{height}`.
"""

        full_prompt = ""
        if rag_examples:
            full_prompt += "Here are some examples of successful past operations. Use them to learn the format.\n\n"
            for example in rag_examples:
                full_prompt += f"User Command: {example['prompt']}\nCode:\n```python\n{example['code']}\n```\n\n"

        full_prompt += f"Now, complete the following task.\nUser Command: {user_prompt}\nCode:\n"

        try:
            print(f"Generating code with model '{self.operation_model}' for prompt: {user_prompt}")
            response = self.client.generate(
                model=self.operation_model,
                system=system_prompt,
                prompt=full_prompt,
                images=[screenshot_path]
            )

            # Clean up the response to get only the code
            generated_code = response['response'].strip()
            if generated_code.startswith("```python"):
                generated_code = generated_code[len("```python"):].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-len("```")].strip()

            print(f"Generated Code:\n{generated_code}")
            return generated_code
        except Exception as e:
            print(f"Error generating code from Ollama: {e}")
            return ""

    def evaluate_operation(self, user_prompt: str, executed_code: str, before_screenshot_path: str, after_screenshot_path: str) -> bool:
        """
        Evaluates if the operation was successful by comparing before and after screenshots.
        """
        print("Evaluating operation with LLM...")
        system_prompt = """
You are a meticulous quality assurance expert. Your task is to determine if a desktop operation was successful.
You will be given:
1. The user's original command.
2. The Python code that was executed.
3. A "before" screenshot taken before the code was executed.
4. An "after" screenshot taken after the code was executed.

Your job is to analyze all of this information to decide if the operation was a success. The executed code is the most important piece of evidence. The "after" screenshot should reflect the result of running that code.
For example, if the code was `pyautogui.click(x=100, y=200)`, the "after" screenshot should show a click or its effect at those coordinates.

Respond with only the word "SUCCESS" or "FAILURE". Do not provide any explanation.
"""

        prompt = f"User Command: {user_prompt}\nExecuted Code:\n```python\n{executed_code}\n```"

        try:
            response = self.client.generate(
                model=self.evaluation_model,
                system=system_prompt,
                prompt=prompt,
                images=[before_screenshot_path, after_screenshot_path]
            )

            result = response['response'].strip().upper()
            print(f"LLM Evaluation Result: {result}")

            return "SUCCESS" in result
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            # In case of evaluation error, assume failure to be safe.
            return False

    def abstract_user_prompt(self, user_prompt: str) -> str:
        """
        Uses an LLM to convert a specific user prompt into a generalized version.
        Example: "Type 'Hello World' in the notepad" -> "Type text into a window"
        """
        print(f"Abstracting user prompt: {user_prompt}")
        system_prompt = """
You are an expert in summarizing user commands. Your task is to convert a specific user command into a general, abstract version.
Focus on the *action* and the *type of target*, removing any specific, literal values like text, numbers, or file names.

Examples:
- User Command: "Click on the 'File' menu" -> Abstract Command: "Click on a menu item"
- User Command: "Type 'Hello World' into the text editor" -> Abstract Command: "Type text into a text editor"
- User Command: "Delete the file named 'report_2024.docx'" -> Abstract Command: "Delete a file"
- User Command: "Move the mouse to coordinates 250, 500" -> Abstract Command: "Move the mouse to a coordinate"
- User Command: "In the 'Sales' spreadsheet, enter '5000' in cell B2" -> Abstract Command: "Enter a value into a cell in a spreadsheet"

Respond with only the abstract command.
"""

        try:
            response = self.client.generate(
                model=self.operation_model, # Use the powerful model for this
                system=system_prompt,
                prompt=user_prompt
            )
            abstracted_prompt = response['response'].strip()
            print(f"Abstracted Prompt: {abstracted_prompt}")
            return abstracted_prompt
        except Exception as e:
            print(f"Error abstracting prompt: {e}. Falling back to original prompt.")
            return user_prompt