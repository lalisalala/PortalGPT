import faiss
import numpy as np
import json
import logging
import os
import sys
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ✅ Add backend folder to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import paths for FAISS index, dataset mappings, and reranker
from config import FAISS_INDEX_PATH, DATASET_MAPPINGS_PATH, DATASETS_JSON_PATH

# ✅ Ensure Python finds the reranker module
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from reranker import rerank  # ✅ Import lightweight reranker

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load FAISS index
logging.info("🔄 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

# ✅ Load dataset ID mappings
with open(DATASET_MAPPINGS_PATH, "r", encoding="utf-8") as f:
    dataset_ids = json.load(f)

# ✅ Load full dataset metadata for enriched search results
with open(DATASETS_JSON_PATH, "r", encoding="utf-8") as f:
    dataset_metadata = {ds["id"]: ds for ds in json.load(f)}

# ✅ Load embedding model (same as indexing)
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-large-en")

# ✅ Create logs directory if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def get_embedding(text):
    """Generate L2-normalized embeddings using the selected model."""
    if not text or text.strip() == "":
        return np.zeros(EMBEDDING_MODEL.get_sentence_embedding_dimension(), dtype=np.float32)
    embedding = EMBEDDING_MODEL.encode(text, normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)

def save_log(query, results):
    """Save search results in a compact format, including both FAISS and reranked rankings."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(LOGS_DIR, f"search_log_{timestamp}.json")

    log_data = {
        "query": query,
        "results": results  # ✅ Already includes both FAISS & Rerank info
    }

    with open(log_filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    logging.info(f"📄 Search results logged in {log_filename}")

def search_faiss(query, k=50):
    """Search FAISS index and return full metadata for top-k results."""
    logging.info(f"🔍 Searching FAISS for query: {query}")

    query_embedding = get_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    # ✅ Collect FAISS results with rankings
    faiss_results = []
    for faiss_rank, (i, d) in enumerate(zip(indices[0], distances[0]), start=1):
        dataset_id = dataset_ids[i]
        dataset_info = dataset_metadata.get(dataset_id, {})

        # ✅ Extract dataset formats (handle multiple formats)
        resources = dataset_info.get("resources", [])
        all_formats = list(set(res.get("format", "Unknown Format").strip() for res in resources if res.get("format")))

        # ✅ Extract dataset landing page (new!)
        landing_page = dataset_info.get("landing_page", "No landing page available")

        # ✅ Extract all download links (new!)
        download_links = [
            {
                "title": res.get("title", "Unnamed Resource"),
                "url": res.get("url", ""),
                "format": res.get("format", "Unknown Format"),
                "size": res.get("size", "Unknown Size"),
                "mimetype": res.get("mimetype", "Unknown MIME Type")
            }
            for res in resources if res.get("url")  # Ensure URL exists
        ]

        faiss_results.append({
            "id": dataset_id,
            "title": dataset_info.get("title", "Unknown Title"),
            "summary": dataset_info.get("summary", "No summary available"),
            "publisher": dataset_info.get("publisher", "Not specified"),
            "tags": dataset_info.get("tags", []),
            "geospatial_coverage": dataset_info.get("geospatial_coverage", {"bounding_box": "Unknown", "smallest_geography": "Unknown"}),
            "temporal_coverage": {
                "from": dataset_info.get("temporal_coverage_from", "Unknown"),
                "to": dataset_info.get("temporal_coverage_to", "Unknown"),
            },
            "format": all_formats if all_formats else ["Unknown Format"],
            "landing_page": landing_page,  # ✅ Include landing page
            "download_links": download_links,  # ✅ Include full download links
            "faiss_score": round(float(d), 4),
            "faiss_rank": faiss_rank  # ✅ Track FAISS ranking
        })

    logging.info(f"✅ Retrieved {len(faiss_results)} results from FAISS.")

    # ✅ Rerank using Lightweight Reranker
    reranked_results = rerank(query, faiss_results)

    # ✅ Assign reranked position
    for rerank_rank, item in enumerate(reranked_results, start=1):
        item["rerank_rank"] = rerank_rank  # ✅ Add reranked position

    # ✅ Save compact log
    save_log(query, reranked_results)

    return reranked_results

# ✅ Run test search if executed directly
if __name__ == "__main__":
    test_query = "I am looking for datasets about homelessness"
    results = search_faiss(test_query, k=50)
    print("🔍 Final Compact Results:", json.dumps(results, indent=4, ensure_ascii=False))