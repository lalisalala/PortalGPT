import os

# Replace 'YOUR_API_KEY' with your actual API key from the London Datastore
LONDON_DATASTORE_API_KEY = os.getenv("LONDON_DATASTORE_API_KEY", "c535ba42-c70c-4cf0-826c-6a93bc6b1f2c ")

# Paths to data files
DATASETS_JSON_PATH = "backend/data/london_datasets.json"
FAISS_INDEX_PATH = "backend/data/dataset_index.faiss"
DATASET_MAPPINGS_PATH = "backend/data/dataset_mappings.json"
