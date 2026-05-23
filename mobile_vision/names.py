import json

with open('class_names.json', 'r') as f:
    categories = json.load(f)

# The categories array inside datasets.INaturalist contains lists/tuples of metadata
# Extract the human-readable string name or taxonomy name at index 1 or 2 depending on version
with open('labels.txt', 'w') as f:
    for cat in categories:
        # Pulling category identifier label name string
        name = cat[1] if isinstance(cat, list) else str(cat)
        f.write(f"{name}\n")

print("Created clean labels.txt for asset embedding.")