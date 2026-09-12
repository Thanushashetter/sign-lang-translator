import json

INPUT_FILE = "data/raw/WLASL_v0.3.json"
OUTPUT_FILE = "shared/gloss_vocab.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

glosses = sorted(item["gloss"].upper() for item in data)

vocab = {
    str(i): gloss
    for i, gloss in enumerate(glosses)
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=4)

print("Vocabulary created successfully!")
print("Number of classes:", len(vocab))
print("Saved to:", OUTPUT_FILE)