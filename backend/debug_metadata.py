from utils.extract_metadata import fetch_metadata

# 🔎 Debugging a single dataset
dataset_id = "electricity-consumption-borough"
metadata = fetch_metadata(dataset_id)

print("\n📌 FINAL EXTRACTED METADATA:")
print(metadata)
