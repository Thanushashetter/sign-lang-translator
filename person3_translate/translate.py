from typing import List

from .llm_client import get_sentence
from .postprocess import clean_sentence


def translate_glosses(glosses: List[str]) -> str:
    sentence = get_sentence(glosses)

    sentence = clean_sentence(sentence)

    return sentence