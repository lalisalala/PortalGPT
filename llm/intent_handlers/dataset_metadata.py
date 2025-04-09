import logging
import os
import subprocess
import json
import textwrap
from llm.session_manager import get_user_session
from backend.utils.retrieval import search_faiss

# ✅ Prompt path setup
PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts"))
METADATA_PROMPT_PATH = os.path.join(PROMPTS_DIR, "dataset_metadata.json")

def load_prompt():
    """Loads the dataset metadata prompt template."""
    if os.path.exists(METADATA_PROMPT_PATH):
        with open(METADATA_PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt_template", "")
    
    logging.warning("⚠️ dataset_metadata.json not found. Using fallback prompt.")
    return (
        "The user asked: \"{query}\".\n\n"
        "Here is metadata from the most recently shown datasets:\n\n{dataset_info}\n\n"
        "Answer the question clearly using only the provided dataset information. "
        "If the user asks about download or documentation, refer them to the dataset's landing page."
    )

# ✅ Updated Keywords – links now mapped to landing page
FIELD_KEYWORDS = {
    "license": ["license", "licencing", "licensing", "terms of use", "usage rights"],
    "publisher": ["publisher", "published by", "owner", "maintainer", "organisation"],
    "format": ["format", "file type", "type of file", "file format"],
    "landing_page": [
        "download", "get the data", "data link", "spreadsheet", "zip file",
        "link", "downloadable", "documentation", "landing page", "more info",
        "source page", "details", "metadata", "learn more", "access", "where can I find"
    ],
    "metadata_dates": ["last updated", "when was this", "created", "updated", "modified", "last modified", "date"]
}

def extract_relevant_fields(query):
    q = query.lower()
    for field, keywords in FIELD_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return [field]
    return ["all"]

def format_all_metadata(datasets, fields=["all"]):
    """Formats dataset metadata for the LLM, with scoped field filtering."""
    formatted_blocks = []

    for i, ds in enumerate(datasets, 1):
        lines = [f"Dataset {i}:"]
        if "title" in fields or "all" in fields:
            lines.append(f"• Title: {ds.get('title', 'Unknown')}")
        if "publisher" in fields or "all" in fields:
            lines.append(f"• Publisher: {ds.get('publisher', 'Unknown Publisher')}")
        if "license" in fields or "all" in fields:
            lines.append(f"• License: {ds.get('license', 'Unknown License')}")
        if "format" in fields or "all" in fields:
            formats = ", ".join(ds.get("format", ["Unknown"])).upper()
            lines.append(f"• Formats: {formats}")
        if "metadata_dates" in fields or "all" in fields:
            lines.append(f"• Created: {ds.get('metadata_created', 'Unknown')}")
            lines.append(f"• Last Updated: {ds.get('metadata_modified', 'Unknown')}")
        if "landing_page" in fields or "all" in fields:
            lines.append(f"• Landing Page: {ds.get('landing_page', 'N/A')}")

        formatted_blocks.append("\n".join(lines))

    return "\n\n".join(formatted_blocks)

def handle_dataset_metadata(query, user_id):
    """Main handler for answering dataset metadata questions."""
    session = get_user_session(user_id)
    datasets = session.get("results", [])[:session.get("shown_count", 5)]

    if not datasets:
        logging.warning(f"⚠️ No shown datasets found for user {user_id}. Using fallback FAISS search.")
        datasets = search_faiss(query, k=5)
        session["results"] = datasets
        session["shown_count"] = 5

    fields = extract_relevant_fields(query)
    dataset_info = format_all_metadata(datasets, fields=fields)
    prompt_template = load_prompt()
    full_prompt = prompt_template.format(query=query, dataset_info=dataset_info)

    logging.info("\n📝 Final Prompt Sent to LLM (Metadata Intent):\n" + full_prompt)

    try:
        result = subprocess.run(
            ["ollama", "run", "llama2:13b", full_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "⚠️ Error generating response from LLM."
    except Exception as e:
        logging.error(f"⚠️ Exception in handle_dataset_metadata: {e}")
        return "⚠️ Something went wrong while processing your request."
