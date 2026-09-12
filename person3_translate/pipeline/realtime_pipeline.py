from typing import List

from person3_translate.translate import translate_glosses


def main():
    # Example output from Person 2's gloss recognition model
    glosses: List[str] = ["STORE", "I", "GO"]

    # Translate glosses into natural English
    sentence = translate_glosses(glosses)

    print("Recognized Glosses:", glosses)
    print("Translated Sentence:", sentence)


if __name__ == "__main__":
    main()