import os
import sys
import json
import logging
import subprocess
import uuid

# ✅ Add the project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Now import the module correctly
from llm.intent_classifier import classify_intent
from llm.intent_handlers.dataset_search import handle_dataset_search 
from llm.intent_handlers.dataset_more_results import handle_dataset_more_results
from llm.session_manager import update_user_history, generate_user_id
from llm.intent_handlers.dataset_metadata import handle_dataset_metadata  
from llm.session_manager import get_user_session

# ✅ Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Path to prompts directory
PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
DATASET_PROMPT_PATH = os.path.join(PROMPTS_DIR, "dataset_search.json")  # 🔹 Use existing dataset search prompt

# ✅ Load intent-specific prompt templates
def load_prompt(intent):
    """Loads the appropriate prompt template from JSON file."""
    prompt_path = os.path.join(PROMPTS_DIR, f"{intent}.json")
    
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt_template", "")
    
    logging.warning(f"⚠️ Prompt file for intent '{intent}' not found. Using fallback.")
    return "The user asked: '{query}'. Provide the best response based on available information."

def generate_response(query, user_id):
    """Processes user query, classifies intent, retains context, and generates response."""

    session = get_user_session(user_id)
    update_user_history(user_id, query)

    raw_intent = classify_intent(query, user_id, use_memory=True)
    logging.info(f"🎯 LLM classified intent: {raw_intent}")
    logging.info(f"🧠 Session has_searched: {session.get('has_searched')}")

    if raw_intent == "dataset_search" and session.get("has_searched", False):
        logging.info("🔁 Already searched. Overriding intent to dataset_metadata.")
        intent = "dataset_metadata"
    else:
        intent = raw_intent

    logging.info(f"🔍 Detected Intent: {intent}")

    # ✅ Route to correct handler or handle unimplemented ones
    if intent == "dataset_search":
        return handle_dataset_search(query, user_id)
    elif intent == "dataset_more_results":
        return handle_dataset_more_results(user_id)
    elif intent == "dataset_metadata":
        return handle_dataset_metadata(query, user_id)
    elif intent == "dataset_explanation":
        return "🔎 Explanation requests are not yet implemented."
    elif intent == "fallback":
        return "💬 This seems like a casual or unrelated query. Let's focus on datasets!"
    else:
        return "🤖 Sorry, I didn't understand that request."


    
if __name__ == "__main__":
    user_id = generate_user_id()  # ✅ Generate unique session ID
    print("🤖 AI: Hello! How can I help you today?")

    while True:
        user_input = input("👤 You: ")  # Take user input
        if user_input.lower() in ["exit", "quit"]:
            print("🤖 AI: Goodbye!")
            break  # ✅ Exit loop

        response = generate_response(user_input, user_id)  # ✅ Pass user_id for session tracking
        print(f"🤖 AI: {response}")