import easyocr
import pyautogui
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import platform
import os

# Initialize the EasyOCR reader with English and Japanese
print("Initializing EasyOCR reader for English and Japanese...")
reader = easyocr.Reader(['en', 'ja'])
print("EasyOCR reader initialized.")

def _get_os_specific_font_path() -> str:
    """
    Determines the appropriate font path based on the operating system.
    This is necessary to handle Japanese characters correctly in the visualization.

    Returns:
        str: The path to a font file that supports Japanese.
    """
    os_name = platform.system()
    if os_name == "Windows":
        # On Windows, Meiryo or MS Gothic are common.
        font_path = "C:\\Windows\\Fonts\\meiryo.ttc"
        if not os.path.exists(font_path):
            font_path = "C:\\Windows\\Fonts\\msgothic.ttc"
        if not os.path.exists(font_path):
            print("Warning: Neither Meiryo nor MS Gothic font found on Windows. Defaulting to a generic path.")
            return "arial.ttf" # Fallback, might not support Japanese
        return font_path
    elif os_name == "Darwin": # macOS
        # On macOS, Hiragino is standard.
        return "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
    else: # Assume Linux
        # On Linux, IPA fonts are a common choice.
        return "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"

# Define the path to a Japanese font file using the helper function
FONT_PATH = _get_os_specific_font_path()


def find_text_coordinates(text_to_find: str) -> Optional[Tuple[int, int]]:
    """
    Finds the coordinates of a given text string on the screen using OCR.

    Args:
        text_to_find (str): The text to search for on the screen.

    Returns:
        Optional[Tuple[int, int]]: The (x, y) coordinates of the center of the
                                   found text box, or None if the text is not found.
    """
    print(f"OCR: Searching for text '{text_to_find}' on screen...")
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    results = reader.readtext(screenshot_np, detail=1, paragraph=False)
    print(f"OCR: Found {len(results)} text block(s).")

    for (bbox, text, confidence) in results:
        print(f"OCR: Checking text '{text}' with confidence {confidence:.2f}")
        if text_to_find.lower().strip() in text.lower().strip():
            print(f"OCR: Found a match for '{text_to_find}'!")
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            print(f"OCR: Bounding box: {bbox}")
            print(f"OCR: Calculated center: ({center_x}, {center_y})")
            return (center_x, center_y)

    print(f"OCR: Text '{text_to_find}' not found on the screen.")
    return None

def get_all_ocr_results(image_path: Optional[str] = None) -> (Image.Image, List[dict]):
    """
    Captures the screen or loads an image from a path and gets all OCR results.

    Args:
        image_path (Optional[str]): The path to an image file. If None, a new screenshot is taken.

    Returns:
        tuple[Image.Image, List[dict]]: A tuple containing the screenshot as a PIL Image
                                 and a list of all detected text blocks.
    """
    if image_path:
        print(f"OCR: Loading image from {image_path} and getting all text...")
        screenshot_image = Image.open(image_path)
    else:
        print("OCR: Capturing screen and getting all text...")
        screenshot_image = pyautogui.screenshot()

    screenshot_np = np.array(screenshot_image)
    results = reader.readtext(screenshot_np, detail=1, paragraph=False)
    print(f"OCR: Found {len(results)} text block(s).")
    # The returned list now only contains the text, not the bbox or confidence
    return screenshot_image, [(bbox, text, confidence) for (bbox, text, confidence) in results]


def draw_ocr_results(screenshot_image: Image.Image, ocr_results: List[dict]) -> Image.Image:
    """
    Draws bounding boxes and text from OCR results onto a screenshot.
    This is a visual debugging tool.

    Args:
        screenshot_image (Image.Image): The original screenshot as a PIL Image.
        ocr_results (List[dict]): A list of OCR results from `get_all_ocr_results`.

    Returns:
        Image.Image: The screenshot with OCR results drawn on it.
    """
    # Ensure the image is in RGBA mode to support transparency
    vis_image = screenshot_image.convert("RGBA")
    draw = ImageDraw.Draw(vis_image)

    try:
        font = ImageFont.truetype(FONT_PATH, 15)
    except IOError:
        print(f"Warning: Font file not found at {FONT_PATH}. Using default font. Japanese characters may not render correctly.")
        font = ImageFont.load_default()

    for (bbox, text, confidence) in ocr_results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Draw the bounding box
        draw.rectangle([top_left, bottom_right], outline="red", width=2)

        # Draw the text slightly above the bounding box
        text_position = (top_left[0], top_left[1] - 20)
        # Add a semi-transparent background for the text using a tuple for color
        text_bbox = draw.textbbox(text_position, text, font=font)
        draw.rectangle(text_bbox, fill=(0, 0, 0, 128)) # Use a tuple for RGBA color
        draw.text(text_position, text, fill="yellow", font=font)

    return vis_image