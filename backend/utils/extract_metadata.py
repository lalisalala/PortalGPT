import requests
import json
import os
import logging
import time
from bs4 import BeautifulSoup 

# Ensure logs directory exists
LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
LOG_FILE_PATH = os.path.join(LOGS_DIR, "extract_metadata.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE_PATH)]
)
logger = logging.getLogger(__name__)

# ✅ Correct API Endpoints
BASE_URL = "https://data.london.gov.uk/api/"
PACKAGE_LIST_URL = BASE_URL + "3/action/package_list"
PACKAGE_SHOW_URL = BASE_URL + "3/action/package_show"

HEADERS = {"User-Agent": "PortalGPT/1.0"}
def clean_html(text):
    """Remove HTML tags from text."""
    return BeautifulSoup(text, "html.parser").get_text()

def fetch_dataset_list():
    """Fetch the list of all dataset identifiers from London Datastore."""
    try:
        response = requests.get(PACKAGE_LIST_URL, headers=HEADERS)
        response.raise_for_status()
        dataset_list = response.json().get("result", [])
        logger.info(f"✅ Fetched {len(dataset_list)} dataset identifiers.")
        return dataset_list
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching dataset list: {e}")
        return []

def fetch_metadata(dataset_id):
    """Fetch all metadata fields for a specific dataset from its JSON URL."""
    try:
        # ✅ Use dataset-specific JSON URL
        url = f"{BASE_URL}dataset/{dataset_id}"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        
        metadata = response.json()  # No need to extract "result" key

        # ✅ Debugging: Print metadata to confirm "smallest_geography" exists
        print(f"\n✅ Metadata for {dataset_id}:")
        print(json.dumps(metadata, indent=4, ensure_ascii=False))
        
        return metadata
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error fetching metadata for dataset {dataset_id}: {e}")
        return {}

def preprocess_metadata(metadata):
    """Preprocess metadata and extract relevant fields, including temporal coverage and smallest geography."""
    if not metadata:
        return {}

    resources = []
    temporal_coverage_from = None
    temporal_coverage_to = None
    smallest_geography = None
    dataset_slug = metadata.get("slug", "")  # Get dataset slug

    # ✅ Extract "smallest geography" from main metadata or "extras"
    smallest_geography = metadata.get("london_smallest_geography", None)
    if not smallest_geography and "extras" in metadata:
        smallest_geography = metadata.get("extras", {}).get("smallest_geography", "Not specified")

    # ✅ Extract geospatial coverage (Bounding Box & Smallest Geography)
    geospatial_coverage = {
        "bounding_box": metadata.get("london_bounding_box", "Unknown bounding box"),
        "smallest_geography": smallest_geography
    }

    # ✅ Extract dataset resources correctly
    if isinstance(metadata.get("resources"), dict):
        for res_id, res in metadata["resources"].items():
            # ✅ Extract the real downloadable link
            corrected_url = f"https://data.london.gov.uk/download/{dataset_slug}/{res_id}/{res['title'].replace(' ', '%20')}"
            
            resources.append({
                "title": res.get("title", "Unnamed Resource"),
                "url": corrected_url,  # ✅ Corrected URL
                "format": res.get("format", "Unknown format"),
                "temporal_coverage_from": res.get("temporal_coverage_from"),
                "temporal_coverage_to": res.get("temporal_coverage_to"),
                "size": res.get("check_size", "Unknown size"),
                "mimetype": res.get("check_mimetype", "Unknown mimetype"),
            })

    
    # ✅ Check for the full license title in "readonly.licence.title" first
    licence_data = metadata.get("readonly", {}).get("licence", {}).get("title")

    # ✅ Construct dataset landing page URL
    dataset_landing_page = f"https://data.london.gov.uk/dataset/{metadata.get('id', '')}"

    # ✅ If not found, fallback to the short licence string
    if not licence_data:
        licence_data = metadata.get("licence", "Unknown License")

    return {
        "id": metadata.get("id", "unknown"),
        "title": metadata.get("title", "Unnamed Dataset"),
        "summary": clean_html(metadata.get("description", "No summary available")),
        "publisher": metadata.get("maintainer", "Unknown Publisher"),
        "tags": metadata.get("tags", []),
        "metadata_created": metadata.get("createdAt", "Unknown Date"),
        "metadata_modified": metadata.get("updatedAt", "Unknown Date"),
        "temporal_coverage_from": temporal_coverage_from,
        "temporal_coverage_to": temporal_coverage_to,
        "geospatial_coverage": geospatial_coverage, 
        "resources": resources,
        "license": licence_data,
        "landing_page": dataset_landing_page # ✅ Directly extract the full license title
    }

def format_for_faiss(metadata_list):
    """Format dataset metadata to be optimized for FAISS embeddings."""
    formatted_data = []
    
    for dataset in metadata_list:
        # 🔹 Prioritize Title & Summary (Most Informative)
        title = dataset.get("title", "Unknown Title")
        summary = dataset.get("summary", "No summary available")
        publisher = dataset.get("publisher", "Unknown Publisher")
        tags = ", ".join(dataset.get("tags", [])) if dataset.get("tags") else "No tags"

        # 🔹 Ensure temporal & geospatial coverage is properly handled
        temporal_from = dataset.get("temporal_coverage_from", "Unknown")
        temporal_to = dataset.get("temporal_coverage_to", "Unknown")
        geospatial_coverage = dataset.get("geospatial_coverage", {}).get("smallest_geography", "Unknown geography")

        # 🔹 Construct Embedding Text (Without Repetitions)
        embedding_text = (
            f"Title: {title}. "
            f"Summary: {summary}. "
            f"Published by {publisher}. "
            f"Tags: {tags}. "
            f"Temporal Coverage: {temporal_from} - {temporal_to}. "
            f"Geospatial Coverage: {geospatial_coverage}."
        )

        formatted_data.append({
            "id": dataset["id"],
            "embedding_text": embedding_text
        })
    
    return formatted_data




def fetch_and_save_metadata(output_file="backend/data/london_datasets.json", faiss_file="backend/data/london_datasets_faiss.json"):
    """Fetch metadata for ALL datasets and save it as a JSON file."""
    dataset_list = fetch_dataset_list()
    if not dataset_list:
        logger.error("❌ No datasets found. Exiting.")
        return

    all_metadata = []
    
    for idx, dataset_id in enumerate(dataset_list):
        logger.info(f"🔍 Fetching metadata for dataset {idx + 1}/{len(dataset_list)}: {dataset_id}")
        metadata = fetch_metadata(dataset_id)
        if metadata:
            processed_metadata = preprocess_metadata(metadata)
            all_metadata.append(processed_metadata)
        time.sleep(0.1)  # Delay to avoid rate limiting

    # ✅ Save metadata in standard format
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=4, ensure_ascii=False)
    logger.info(f"✅ Metadata saved to '{output_file}'")

    # ✅ Save FAISS-formatted data
    faiss_ready_data = format_for_faiss(all_metadata)
    with open(faiss_file, "w", encoding="utf-8") as f:
        json.dump(faiss_ready_data, f, indent=4, ensure_ascii=False)
    logger.info(f"✅ FAISS formatted data saved to '{faiss_file}'")

if __name__ == "__main__":
    fetch_and_save_metadata()