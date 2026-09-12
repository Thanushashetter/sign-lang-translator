from person3_translate.llm_client import get_sentence


glosses = ["STORE", "I", "GO"]

sentence = get_sentence(glosses)

print("Glosses:", glosses)
print("Sentence:", sentence)