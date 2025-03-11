import faiss
import numpy as np
import json
import os
import sys
import logging
from sentence_transformers import SentenceTransformer  # Open-source embeddings

# Ensure logs directory exists
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
LOG_FILE_PATH = os.path.join(LOGS_DIR, "faiss_index.log")
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s", 
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE_PATH)]
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Ensure paths are correct
from config import FAISS_INDEX_PATH, DATASET_MAPPINGS_PATH

# ✅ Load FAISS-ready dataset JSON
FAISS_READY_JSON_PATH = "backend/data/london_datasets_faiss.json"

# Load embedding model
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-large-en")

def get_embedding(text):
    """Generate normalized embeddings using the selected model."""
    if not text or text.strip() == "":
        return np.zeros(EMBEDDING_MODEL.get_sentence_embedding_dimension(), dtype=np.float32)
    return np.array(EMBEDDING_MODEL.encode(text, normalize_embeddings=True), dtype=np.float32)

def build_faiss_index():
    """Create FAISS index from structured metadata."""
    logging.info("🔍 Loading FAISS-ready dataset metadata...")

    # ✅ Load FAISS formatted JSON file
    with open(FAISS_READY_JSON_PATH, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    if not datasets:
        logging.error("❌ No dataset metadata found! Run extract_metadata.py first.")
        return

    embeddings = []
    dataset_ids = []

    for ds in datasets:
        dataset_id = ds["id"]
        embedding_text = ds.get("embedding_text", "")

        # ✅ Generate the embedding
        embedding = get_embedding(embedding_text)

        embeddings.append(embedding)
        dataset_ids.append(dataset_id)

    # Convert embeddings to NumPy array
    embeddings_np = np.array(embeddings, dtype=np.float32)

    # ✅ Normalize for cosine similarity
    faiss.normalize_L2(embeddings_np)

    # ✅ Ensure FAISS index uses correct dimensions
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    # ✅ Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # ✅ Save dataset ID mappings
    with open(DATASET_MAPPINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_ids, f, indent=4)

    logging.info(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")
    logging.info(f"✅ Dataset mappings saved to {DATASET_MAPPINGS_PATH}")

if __name__ == "__main__":
    build_faiss_index()
