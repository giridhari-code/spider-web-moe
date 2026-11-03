"""Product vs architecture naming.

Model (product):  Sentinel
Architecture:     Spider (Spider-Web Looped MoE)

Same pattern as Grok (model) vs the underlying research stack.
"""

MODEL_NAME = "Sentinel"
MODEL_VERSION = "1.0"
MODEL_ID = "sentinel"
ARCHITECTURE_NAME = "Spider"
ARCHITECTURE_FULL = "Spider-Web Looped MoE"

def display_name() -> str:
    return f"{MODEL_NAME} {MODEL_VERSION}"

def long_description() -> str:
    return f"{MODEL_NAME} {MODEL_VERSION} — {ARCHITECTURE_FULL} architecture"
