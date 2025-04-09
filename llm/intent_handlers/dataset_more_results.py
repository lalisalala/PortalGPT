import logging
import subprocess
from llm.intent_handlers.dataset_search import format_dataset_info
from llm.session_manager import get_next_faiss_results

def handle_dataset_more_results(user_id):
    """Fetches additional datasets beyond the initially retrieved results."""
    
    # ✅ Get the next batch of FAISS results
    next_datasets = get_next_faiss_results(user_id, batch_size=5)

    if not next_datasets:
        return "⚠️ No more datasets available. Try refining your search."

    dataset_titles = [dataset["title"] for dataset in next_datasets]
    logging.info(f"🔍 Next batch of datasets for user {user_id}: {dataset_titles}")

    # ✅ Format the new dataset batch
    dataset_info = format_dataset_info(next_datasets)

    # ✅ Construct a natural response prompt for LLM
    prompt = f"""
    The user requested additional datasets. Below are the next available datasets:

    {dataset_info}

    **IMPORTANT:** These are new datasets, not the ones previously shown. Ensure the response acknowledges that these are additional results.
    """

    logging.info("\n📝 **Final Prompt Sent to LLM (More Results Intent):**\n" + prompt)

    # ✅ Call the LLM to process and refine the response
    try:
        result = subprocess.run(
            ["ollama", "run", "llama2:13b", prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "⚠️ Error generating response."
    
    except Exception as e:
        logging.error(f"⚠️ Exception in LLM response: {str(e)}")
        return "⚠️ Sorry, something went wrong."
