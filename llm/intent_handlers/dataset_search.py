import logging
import os
import json
import subprocess
import textwrap
from backend.utils.retrieval import search_faiss  # ✅ Use FAISS retrieval from `retrieval.py`
from llm.session_manager import store_faiss_results, get_user_session 

# ✅ Path to prompts directory
PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "prompts"))

def load_prompt():
    """Loads the dataset search prompt from dataset_search.json."""
    prompt_path = os.path.join(PROMPTS_DIR, "dataset_search.json")

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt_template", "")

    logging.warning(f"⚠️ dataset_search.json not found. Using fallback prompt.")
    return "The user is looking for datasets related to: \"{query}\". {dataset_info}"

def summarize_text(dataset):
    """Generates a concise, optimized 1-2 sentence summary using available metadata."""

    title = dataset.get("title", "Unknown Title")
    description = dataset.get("summary", "").strip()
    publisher = dataset.get("publisher", "Unknown Publisher")
    formats = ", ".join(dataset.get("format", ["Unknown Format"])).upper()
    tags = ", ".join(dataset.get("tags", [])) if dataset.get("tags") else "No Tags"
    
    # ✅ Handle missing temporal coverage
    temporal_coverage = dataset.get("temporal_coverage", {})
    temporal_from = temporal_coverage.get("from", "Unknown") or "Unknown"
    temporal_to = temporal_coverage.get("to", "Unknown") or "Unknown"
    time_coverage = f"{temporal_from} - {temporal_to}" if temporal_from != "Unknown" or temporal_to != "Unknown" else "Unknown"

    # ✅ Build optimized summarization prompt
    metadata_prompt = textwrap.dedent(f"""
        Create a **concise, 2-3 sentence summary** for the following dataset using its metadata:
        - **Title:** {title}
        - **Description:** {description if description else 'No description available'}
        - **Publisher:** {publisher}
        - **Formats:** {formats}
        - **Tags:** {tags}
        - **Time Coverage:** {time_coverage}
        
        **ONLY return the summary.** Do NOT include unnecessary instructions or dataset details.
    """).strip()

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", metadata_prompt],
            capture_output=True,
            text=True,
            timeout=20
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logging.warning(f"⚠️ Summarization error: {result.stderr}")
            return description if description else "No summary available."  # Fallback
    except Exception as e:
        logging.error(f"⚠️ Exception during summarization: {str(e)}")
        return description if description else "No summary available."  # Fallback

def format_dataset_info(datasets):
    """Formats dataset information with only title and summary (minimalist mode)."""
    formatted_info = ""

    for i, dataset in enumerate(datasets, 1):
        title = dataset.get("title", "Unknown Title")
        summary = summarize_text(dataset)  # Still uses rich metadata for smart summary

        formatted_info += textwrap.dedent(f"""
        **Dataset {i}**
        **Title:** {title}
        🔹 **Summary:** {summary}
        """).strip() + "\n\n"

    return formatted_info.strip()


def clean_llm_output(response_text):
    """Removes redundant instructions from LLM output."""
    stop_phrases = [
        "Format your response using the structure given in the instructions.",
        "Ensure all datasets are included and formatted correctly.",
        "Please generate a structured response."
    ]
    for phrase in stop_phrases:
        response_text = response_text.split(phrase)[0].strip()
    return response_text

def reformulate_query(query):
    """Refines user queries only if they are vague, otherwise returns as-is."""
    
    # ✅ If the query is already well-structured, return it directly
    if len(query.split()) <= 3:  # Short queries may need reformulation
        reformulation_prompt = f"""
        You are an AI assistant that refines vague queries for dataset retrieval.
        If the user query is already well-structured, return it as-is.
        Otherwise, improve it while keeping it concise and preserving its meaning.

        **User Query:** "{query}"
        
        **Improved Query:** (Return only the refined query, no extra text)
        """

        try:
            result = subprocess.run(
                ["ollama", "run", "mistral", reformulation_prompt],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logging.warning(f"⚠️ Query reformulation failed, using original query.")
                return query  # Fallback to original query

        except Exception as e:
            logging.error(f"⚠️ Exception in query reformulation: {str(e)}")
            return query  # Fallback to original query

    else:
        return query  # ✅ Return original query if it's already detailed


def handle_dataset_search(query, user_id):
    """Handles dataset search queries by refining the query, retrieving FAISS results, and formatting the response for LLM."""
    
    # ✅ Step 1: Reformulate the Query
    refined_query = reformulate_query(query)
    logging.info(f"🔍 Original Query: {query} → Reformulated Query: {refined_query}")

    # ✅ Step 2: Search FAISS with the refined query
    datasets = search_faiss(refined_query, k=50)  
    top_datasets = datasets[:3]  # ✅ First batch of results

    # ✅ Step 3: Store results in session
    session = get_user_session(user_id)
    store_faiss_results(user_id, datasets)

    # 🔥 FIX: Update `shown_count` to 3 (ensures pagination starts at 6 next time)
    session["shown_count"] = 3
    session["has_searched"] = True 
    logging.info(f"✅ Updated `shown_count` to 3 for user {user_id}")

    dataset_info = format_dataset_info(top_datasets)
    dataset_prompt_template = load_prompt()

    # ✅ Step 4: LLM-Powered Response
    full_prompt = f"""
    You are an AI dataset assistant that helps users find and recommend datasets.

    **User Query:** "{query}"
    **Refined Query for Search:** "{refined_query}"

    **Dataset Search Results:**
    {dataset_info}

    Instructions:
    - If the user is asking for a general dataset search, summarize the most relevant datasets.
    - If the user is asking for the "best" dataset, intelligently choose the most relevant dataset and explain why.
    - If multiple datasets could be the best, suggest a few with a brief explanation.

    **AI Response:**
    """

    logging.info("\n📝 **Final Prompt Sent to LLM (Dataset Search Intent):**\n" + full_prompt)

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", full_prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            response_text = result.stdout.strip()
            cleaned_response = clean_llm_output(response_text)  
            return cleaned_response
        else:
            return "⚠️ Error generating response."
    
    except Exception as e:
        logging.error(f"⚠️ Exception in dataset search: {str(e)}")
        return "⚠️ Sorry, something went wrong."
