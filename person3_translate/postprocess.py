def clean_sentence(text: str) -> str:
    text = text.strip()

    if not text:
        return ""

    if text[-1] not in ".!?":
        text += "."

    return text