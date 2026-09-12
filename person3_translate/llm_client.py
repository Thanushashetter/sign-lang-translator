from typing import List
import os
import time

from dotenv import load_dotenv
from google import genai

from .prompt_templates import GLOSS_TO_SENTENCE_PROMPT

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY was not loaded. Check your .env file.")

client = genai.Client()

MODEL_NAME = "gemini-3.8-flash"
MAX_RETRIES = 3


def get_sentence(glosses: List[str]) -> str:

    prompt = GLOSS_TO_SENTENCE_PROMPT.format(
        glosses=glosses
    )

    for attempt in range(MAX_RETRIES):

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:

            if attempt == MAX_RETRIES - 1:
                raise e

            time.sleep(2)