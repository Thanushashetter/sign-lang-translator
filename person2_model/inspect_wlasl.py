import json

FILE = "data/raw/WLASL_v0.3.json"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of glosses:", len(data))

item = data[0]

print("\nFirst gloss:")
print(item["gloss"])

print("\nFirst instance:")
print(item["instances"][0])