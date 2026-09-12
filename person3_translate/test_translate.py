from .translate import translate_glosses


glosses = ["STORE", "I", "GO"]

sentence = translate_glosses(glosses)

print("Glosses:", glosses)
print("Translated sentence:", sentence)