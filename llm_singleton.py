from llm_handler import LLMHandler

_llm_handler_instance = None

def get_llm_handler():
    """
    Returns a single persistent LLMHandler instance.
    Prevents GPU memory from being released even after Streamlit reloads.
    """
    global _llm_handler_instance
    if _llm_handler_instance is None:
        print("[LLM] Initializing persistent handler...")
        _llm_handler_instance = LLMHandler("config.json")

    return _llm_handler_instance
