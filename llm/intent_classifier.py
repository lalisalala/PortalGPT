import subprocess
from llm.session_manager import get_user_session  # ✅ No circular import

# ✅ Updated Intent List (Now Includes "dataset_more_results")
INTENTS = [
    "dataset_search",       # 🔹 Includes dataset recommendations
    "dataset_more_results", # 🔹 Retrieve additional FAISS results (pagination)
    "dataset_metadata",     # 🔹 Fetching dataset details (license, source, etc.)
    "dataset_explanation",  # 🔹 Asking about dataset-related terms (e.g., "What is NO₂?")
    "fallback"              # 🔹 General conversation (e.g., "hi", "How are you?")
]

# ✅ System Prompt for Intent Classification
# ✅ System Prompt for Intent Classification with Few-Shot Examples
SYSTEM_PROMPT = f"""
You are a dataset discovery assistant. Your job is to classify user queries into the most appropriate category.

### **Intent Definitions & Examples:**

#### 🔍 **dataset_search** → The user is looking for datasets or asking for dataset recommendations.
**Examples:**
- "I need datasets on climate change."
- "Do you have data on air pollution in London?"
- "Can you suggest some datasets on housing prices?"
- "What datasets do you have related to public transport?"
- "Give me datasets about population demographics."

#### ➕ **dataset_more_results** → The user wants to see more datasets from the last search.
**Examples:**
- "Show me more datasets."
- "Do you have additional datasets?"
- "I need more results from my previous search."
- "Are there any other datasets available?"
- "Give me a few more options."

#### ℹ️ **dataset_metadata** → The user wants information about a dataset (e.g., format, license, publisher, download link, landing page).
**Examples:**
- "What is the license for these datasets?"
- "Who published this dataset?"
- "What formats is this dataset available in?"
- "When were these datasets last updated?"
- "Is this dataset free to use?"
- "Where can I download these datasets?"
- "Where can I learn more about these datasets"
- "Where is the documentation?"


#### 🤔 **dataset_explanation** → The user is asking about the meaning of a term, abbreviation, or dataset-related concept.
**Examples:**
- "What does NO₂ mean?"
- "Explain what GDP per capita is."
- "What is a shapefile?"
- "Can you tell me what PM2.5 is?"
- "What does CSV mean in datasets?"

#### 💬 **fallback** → The user's message is NOT related to datasets (e.g., greetings, casual conversation).
**Examples:**
- "Hi"
- "How are you?"
- "Tell me a joke!"
- "Who created you?"
- "What's the weather like today?"

---
### **Instructions for Classification:**
1. **Read the user’s message** and classify it into one of these categories.
2. If the message is a **greeting or general conversation**, classify it as `"fallback"`.
3. If the user is requesting **more datasets beyond what was already shown**, classify as `"dataset_more_results"`.
4. If the user is asking about details of previously shown datasets (like format, license, publisher), classify it as `"dataset_metadata"`.
5. If the message is a dataset-related question that involves defining a term or abbreviation (e.g. PM2.5, GDP), classify it as `"dataset_explanation"`.
6. **Respond ONLY with one of these intents:** dataset_search, dataset_more_results, dataset_metadata, dataset_explanation, fallback
"""


def classify_intent(query: str, user_id: str, use_memory: bool = False) -> str:
    """Classify user intent using Mistral via Ollama with session memory."""
    session = get_user_session(user_id)  # 🔧 Add this at the top to access session
    past_context = session["history"] if use_memory else []

    history_str = "\n".join(past_context) if past_context else "No prior context available."

    prompt = f"""
{SYSTEM_PROMPT}

**Conversation History:**
{history_str}

**Latest User Query:**  
{query}

Respond ONLY with one of these intents: {', '.join(INTENTS)}.
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            response = result.stdout.strip().lower()

            # ✅ Safety: make sure it's a known intent
            if response in INTENTS:

                # 🔧 [NEW] Check if FAISS has already run and override if needed
                if response == "dataset_search" and session.get("has_searched", False):
                    logging.info("🔁 FAISS already used — treating query as dataset_metadata instead.")
                    return "dataset_metadata"

                return response

        return "dataset_search"  # Default fallback

    except Exception as e:
        logging.error(f"⚠️ Exception during intent classification: {e}")
        return "dataset_search"