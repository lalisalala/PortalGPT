import logging
import subprocess
from llm.intent_handlers.dataset_search import format_dataset_info
from llm.session_manager import get_user_session

def handle_dataset_filtering(query, user_id):
    """Filters only the datasets that the user has already seen, based on user constraints (e.g., format, time range)."""

    session = get_user_session(user_id)
    all_results = session.get("results", [])  # ✅ Get all FAISS results from session
    shown_count = session.get("shown_count", 0)  # ✅ How many datasets the user has seen

    if not all_results or shown_count == 0:
        logging.warning(f"⚠️ No previously shown datasets to filter for user {user_id}")
        return "⚠️ No datasets to filter. Try searching for datasets first."

    # ✅ Only filter from already SHOWN datasets (not all FAISS results)
    visible_results = all_results[:shown_count]

    # ✅ Extract user filtering preferences using LLM
    filtering_prompt = f"""
    You are an AI assistant that extracts filtering criteria from user queries.

    **User Query:** "{query}"

    **Instructions:**
    - Identify whether the user wants to filter by **format (PDF, CSV, etc.), date range, or topic**.
    - If they mention a format (e.g., "only PDFs"), extract it.
    - If they mention a time range (e.g., "from 2000 to 2010"), extract it.
    - If they mention a topic, return the topic as a refinement.

    **Extracted Filtering Criteria:** (Return only the filtering criteria, no extra text)
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", filtering_prompt],
            capture_output=True,
            text=True,
            timeout=15
        )

        if result.returncode == 0:
            filter_criteria = result.stdout.strip()
            logging.info(f"✅ Extracted Filtering Criteria: {filter_criteria}")
        else:
            return "⚠️ Could not understand filtering request. Try specifying format (e.g., PDF) or date range."

    except Exception as e:
        logging.error(f"⚠️ Exception in filtering extraction: {str(e)}")
        return "⚠️ Could not process filtering request."

    # ✅ Apply filtering ONLY to datasets that the user has ALREADY SEEN
    filtered_results = []

    for dataset in visible_results:
        if "pdf" in filter_criteria.lower():
            if "pdf" not in [fmt.lower() for fmt in dataset.get("format", [])]:
                continue  # Skip if dataset format doesn't match

        filtered_results.append(dataset)

    if not filtered_results:
        return "⚠️ No datasets match your filter criteria. Try adjusting your filters."

    dataset_info = format_dataset_info(filtered_results)
    
    return f"🔍 Here are the datasets matching your filters:\n\n{dataset_info}"
