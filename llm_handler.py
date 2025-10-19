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
        """
        width, height = screen_size
        system_prompt = f"""
You are an expert Python programmer creating a script for desktop automation. Your code must be perfect, robust, and follow best practices.

**1. Tool Selection**
First, determine the correct tool for the job based on the user's prompt.
- For **web browser** tasks (e.g., "Google", "website", "URL"), you **must** use the `playwright` library.
- For **desktop applications** (e.g., "Notepad", "Calculator"), you **must** use `pyautogui`, `pygetwindow`, and `pyperclip`.

**2. Desktop Automation Best Practices (`pyautogui`)**
To ensure your script is reliable, follow this exact workflow:
1.  **Check for Existing Window:** Before launching an app, always check if it's already open using `pygetwindow.getWindowsWithTitle('AppName')`.
2.  **Activate or Launch:** If found, activate the window with `.activate()`. If not found, **and only then**, launch it with `subprocess.Popen(['app.exe'])`. After launching, you must wait (`time.sleep()`) and then get and activate the window.
3.  **Typing Non-ASCII Text:** For any text that is not plain English (e.g., Japanese), you **must** use the clipboard method to avoid character corruption.
    - **Required:** `import pyperclip` at the top of your script.
    - **Required:** Use `pyperclip.copy('こんにちは')` followed by `pyautogui.hotkey('ctrl', 'v')`.
4.  **Literal Values:** Never store literal text in a variable. Embed it directly in the function call (e.g., `pyperclip.copy('some_text')`).

**3. Web Automation Best Practices (`playwright`)**
To create a high-quality Playwright script, you must adhere to these rules:
1.  **Async Boilerplate:** All scripts **must** begin with the standard `nest_asyncio` setup to prevent event loop conflicts.
    ```python
    import asyncio
    import nest_asyncio
    from playwright.async_api import async_playwright

    nest_asyncio.apply()
    ```
2.  **Structure:** All your code must be within an `async def main()` function, called by `asyncio.run(main())`.
3.  **Selector Best Practices (CRITICAL):**
    - **Google Search:** You **MUST** use `page.locator('textarea[name=\"q\"]')`. This is the most reliable selector based on dynamic verification.
    - **Yahoo! JAPAN Search:** You **MUST** use `page.locator('[role=\"search\"] input[type=\"search\"]')`. This has also been dynamically verified as the most stable selector.
    - **General Principle:** For other sites, prefer role-based locators (`get_by_role`, `get_by_label`) over fragile CSS or XPath selectors.
4.  **Keep Browser Open:** To allow for verification, your script **must** end with `await asyncio.Future()`. This will keep the browser window open.
5.  **Correct `await` Usage:** Remember that `await` is only for `async` functions. `page.get_by_role(...)` is NOT async. `await page.get_by_role(...)` is a `TypeError`. The correct pattern is `locator = page.get_by_role(...)` followed by `await locator.fill(...)`.

**4. Final Output**
- You **must** provide only the complete, executable Python code.
- Ensure all necessary libraries (e.g., `pyperclip`, `pygetwindow`, `subprocess`) are imported.

Screen Resolution: `{width}x{height}`.
"""

        full_prompt = ""
        if rag_examples:
            full_prompt += "Here are some examples of successful past operations. Use them as a reference for the correct format and style.\n\n"
            for example in rag_examples:
                full_prompt += f"User Command: {example['prompt']}\nCode:\n```python\n{example['code']}\n```\n\n"

        full_prompt += f"Now, write a Python script that achieves the following goal.\nUser Command: {user_prompt}\nCode:\n"

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
You are a meticulous quality assurance expert. Your task is to determine if a desktop automation operation was successful by comparing "before" and "after" screenshots.

Follow this **Chain of Thought** to make your determination:

1.  **Analyze the User's Goal:** First, understand the user's original command. What was the core intent? (e.g., "type 'hello world'", "open the file menu", "search for 'cats'").

2.  **Identify the Expected Outcome:** Based on the user's goal and the executed code, what is the single most important visual change you expect to see in the "after" screenshot?
    *   *Example for typing:* "The text 'hello world' should appear in a text field."
    *   *Example for clicking:* "A new menu, window, or button state should appear where the click occurred."
    *   *Example for searching:* "The page should show search results related to 'cats'."

3.  **Compare Screenshots for Evidence:** Look for the expected outcome in the "after" screenshot. Is there clear, unambiguous visual evidence that the goal was achieved? The "before" screenshot is for context.

4.  **Final Judgment:** Based on your analysis, conclude with **only** the word "SUCCESS" or "FAILURE". Do not provide any other text or explanation. If the visual evidence is missing or ambiguous, you must conclude "FAILURE".
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