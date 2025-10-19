import pyautogui
import os
import subprocess
import sys
from datetime import datetime
from PIL import Image

class DesktopController:
    """
    Handles direct interactions with the desktop, such as taking screenshots,
    managing the mouse and keyboard, and executing generated code.
    """
    def __init__(self, screenshots_dir: str = "screenshots"):
        """
        Initializes the controller and ensures the screenshot directory exists.
        """
        self.screenshots_dir = screenshots_dir
        os.makedirs(self.screenshots_dir, exist_ok=True)
        print(f"DesktopController initialized. Screenshots will be saved in '{self.screenshots_dir}'.")

    def capture_screenshot(self) -> Image.Image:
        """
        Captures the entire screen and returns it as a PIL Image object.
        """
        print("Capturing screenshot...")
        screenshot = pyautogui.screenshot()
        print("Screenshot captured successfully.")
        return screenshot

    def save_screenshot(self, screenshot_image: Image.Image, prefix: str = "screenshot") -> str:
        """
        Saves a PIL Image object to a file with a timestamp.

        Args:
            screenshot_image (Image.Image): The image to save.
            prefix (str): A prefix for the filename.

        Returns:
            str: The path to the saved screenshot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = os.path.join(self.screenshots_dir, filename)

        print(f"Saving screenshot to {filepath}...")
        screenshot_image.save(filepath)
        print("Screenshot saved successfully.")
        return filepath

    def get_screen_size(self) -> tuple[int, int]:
        """
        Gets the screen resolution.

        Returns:
            A tuple (width, height).
        """
        width, height = pyautogui.size()
        print(f"Screen size detected: {width}x{height}")
        return width, height

    def execute_code(self, code: str):
        """
        Executes a string of Python code in a separate process.
        This is crucial for ensuring that `pyautogui` commands work correctly
        and don't interfere with the main application. It also handles
        non-ASCII characters correctly.

        Args:
            code (str): The Python code to execute.
        """
        print(f"Executing code:\n---\n{code}\n---")

        # Create a temporary file to store the code, ensuring UTF-8 encoding with BOM
        # for compatibility with Windows and non-ASCII characters.
        temp_script_path = "temp_automation_script.py"
        try:
            with open(temp_script_path, 'w', encoding='utf-8-sig') as f:
                f.write(code)

            # Execute the script in a new, independent process
            # Using sys.executable ensures we use the same Python interpreter.
            # Setting PYTHONIOENCODING ensures stderr/stdout are handled correctly.
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Execute the script in a new, non-blocking process using Popen
            # This allows the main app to continue without waiting for the script to finish,
            # which is essential for leaving browser windows open.
            # Redirect stdout and stderr to log files to capture output from the detached process
            with open("script_output.log", "w", encoding='utf-8') as stdout_log, \
                 open("script_error.log", "w", encoding='utf-8') as stderr_log:
                process = subprocess.Popen(
                    [sys.executable, temp_script_path],
                    stdout=stdout_log,
                    stderr=stderr_log,
                    env=env,
                    start_new_session=True
                )

            # We don't wait for the process to complete here.
            # The script will run independently as a detached process.
            print(f"Launched script with PID: {process.pid}")
            print("Execution command sent. The script will run independently.")

        except Exception as e:
            # This will catch errors related to launching the process itself.
            # Errors from within the script will run in the background.
            print(f"Error launching the script with Popen: {e}", file=sys.stderr)
            raise e
        # The temporary script is NOT cleaned up here.
        # It must persist for the subprocess to execute it.
        # Cleanup should be handled by the main app flow after the operation is complete.