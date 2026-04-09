import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

if not COHERE_API_KEY:
    # We will use Gemini in the integrated version, but keeping this for the original file structure
    COHERE_API_KEY = "PLACEHOLDER"
