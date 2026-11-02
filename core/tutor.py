import os
import time

from google import genai
from google.genai import types
from google.genai.errors import APIError

# Valid default model identifier for Gemini 2.5 Flash
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """
You are StudyMind AI, an intelligent personal tutor.

Your responsibilities:
1. Explain concepts in simple language.
2. Teach step by step.
3. Provide practical examples.
4. Adapt explanations to the student's level.
5. Encourage critical thinking.
6. Give accurate and helpful answers.
"""


def _get_client():
    """Dynamically obtain client to pick up runtime environment variables."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is missing. "
            "Set it in Streamlit secrets or OS environment."
        )
    return genai.Client(api_key=api_key)


def _build_contents(question, history):
    contents = []

    for message in history or []:
        role = "user" if message["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message["content"])],
            )
        )

    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        )
    )

    return contents


def ask_tutor(question, history=None):
    contents = _build_contents(question, history)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.4,
    )

    last_error = None

    for attempt in range(3):
        try:
            client = _get_client()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config,
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response "
                    "(it may have been blocked by safety filters)."
                )

            return response.text

        except APIError as error:
            last_error = error
            if error.code in (429, 500, 503) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
        except Exception as error:
            raise error

    raise last_error