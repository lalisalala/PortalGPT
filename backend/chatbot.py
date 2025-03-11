import faiss
import json
import ollama  # Llama2 for NLP
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from bs4 import BeautifulSoup

# ✅ Paths to FAISS index and dataset mappings
FAISS_INDEX_PATH = "backend/data/dataset_index.faiss"
METADATA_PATH = "backend/data/london_datasets.json"
DATASET_MAPPINGS_PATH = "backend/data/dataset_mappings.json"

# ✅ Load embedding model (same as FAISS)
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-large-en")

# ✅ Load dataset metadata
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    dataset_metadata = json.load(f)

# ✅ Load dataset ID mappings
with open(DATASET_MAPPINGS_PATH, "r", encoding="utf-8") as f:
    dataset_mappings = json.load(f)

# ✅ Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)

# ✅ Chat history and session data
chat_history = []
retrieved_datasets = None  # ✅ Stores the latest FAISS search results

# ✅ System prompt for the LLM
SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in datasets and data analysis. "
    "You help users explore, analyze, and find datasets. "
    "You only provide dataset insights when explicitly asked."
)


def clean_html(text):
    """Remove HTML tags and return clean text."""
    if not text:
        return "No summary available."
    return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()

def classify_query_type(user_query):
    """Use LLM to classify whether the query is a dataset search, a follow-up, or general chat."""

    system_prompt = (
        "You are an AI assistant specializing in datasets and natural conversation. "
        "Your task is to determine the intent behind the user's message and classify it correctly. "
        "Use chat history to understand context.\n\n"

        "**Classify the intent as one of the following:**\n"
        "1️⃣ **'dataset_search'** → When the user is asking to find a dataset. (Trigger FAISS search)\n"
        "   - Examples:\n"
        "     - 'Find me data on air pollution.'\n"
        "     - 'Do you have any datasets about traffic patterns in London?'\n"
        "     - 'I'm looking for datasets related to crime statistics.'\n\n"

        "2️⃣ **'dataset_followup'** → When the user is referring to a dataset they have already searched for. (Analyze found datasets)\n"
        "   - Examples:\n"
        "     - 'Tell me more about the second dataset you showed.'\n"
        "     - 'How can I use dataset 1?'\n"
        "     - 'Compare the datasets you just provided.'\n"
        "     - 'Can you summarize that air pollution dataset?'\n\n"

        "3️⃣ **'general_conversation'** → When the user is chatting about a topic unrelated to datasets. (Chat freely with the LLM)\n"
        "   - Examples:\n"
        "     - 'Hey, how are you?'\n"
        "     - 'What is machine learning?'\n"
        "     - 'Tell me a joke!'\n"
        "     - 'Can you explain regression analysis?'\n\n"

        "**Important Notes:**\n"
        "- If the user mentions 'dataset', but it is a request for analysis, classify it as 'dataset_followup'.\n"
        "- If the user asks for a completely new dataset, classify it as 'dataset_search'.\n"
        "- If no dataset has been found in this session, do not classify anything as 'dataset_followup'.\n"
        "- Respond with **only one of these three categories**, no additional text."
    )

    response = ollama.chat(
        model="mistral",  # ✅ Fast model, ideal for classification
        messages=[{"role": "system", "content": system_prompt}]
        + chat_history[-5:]  # Provide recent context for better accuracy
        + [{"role": "user", "content": user_query}]
    )

    return response["message"]["content"].strip().lower()





def search_datasets(query, top_k=10):
    """Search FAISS index for relevant datasets and store them in chat history."""
    global chat_history

    query_embedding = EMBEDDING_MODEL.encode(query, normalize_embeddings=True).reshape(1, -1).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(dataset_mappings):
            dataset_id = dataset_mappings[idx]
            dataset_info = next((d for d in dataset_metadata if d["id"] == dataset_id), None)
            if dataset_info:
                results.append(dataset_info)

    # ✅ Store FAISS results in chat history under "datasets_found"
    chat_history.append({"role": "assistant", "datasets_found": results})

    return results




def format_datasets(results):
    """Format dataset search results for display."""
    return "\n".join(
        [
            f"📌 **{i+1}. {data['title']}**\n"
            f"   🔹 **Summary:** {clean_html(data.get('summary', 'No summary available'))}\n"
            f"   📍 **Geography:** {data.get('smallest_geography', 'N/A')}\n"
            f"   🏛 **Publisher:** {data.get('publisher', 'N/A')}\n"
            f"   📜 **License:** {data.get('license', 'N/A')}\n"
            f"   📂 **Format:** {data['resources'][0]['format'] if data.get('resources') else 'N/A'}\n"
            f"   🔗 **Download:** [Click Here]({data['resources'][0]['url'] if data.get('resources') else 'N/A'})\n"
            f"──────────────────────────────────────────"
            for i, data in enumerate(results)
        ]
    )


def generate_chatbot_response(user_query):
    """Decide whether to trigger FAISS search or let Llama2 handle the response."""
    global chat_history

    query_type = classify_query_type(user_query)

    # 🔍 **If user is searching for datasets, trigger FAISS**
    if query_type == "dataset_search":
        results = search_datasets(user_query)

        if results:
            faiss_response = format_datasets(results)  # ✅ Format for display

            # ✅ Store FAISS results in chat history for follow-ups
            chat_history.append({"role": "assistant", "datasets_found": results})

            return faiss_response  # Show datasets to the user

        return "❌ No relevant datasets found. Try rewording your query."

    # 🔎 **If user is following up on datasets, retrieve stored FAISS results**
    if query_type == "dataset_followup":
        last_datasets = next((msg["datasets_found"] for msg in reversed(chat_history) if "datasets_found" in msg), None)

        if not last_datasets:
            return "I don't have dataset context yet. Try searching for one first!"

        # ✅ Pass FAISS results to Llama2 for deeper analysis
        llm_response = analyze_dataset_with_llm(user_query, last_datasets)
        return llm_response

    # 💬 **If it's general conversation, let Llama2 answer freely**
    response = ollama.chat(
        model="llama2:13b",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."}]
        + chat_history
        + [{"role": "user", "content": user_query}]
    )

    bot_response = response["message"]["content"]
    chat_history.append({"role": "assistant", "content": bot_response})

    return bot_response




def analyze_dataset_with_llm(user_query, previous_results):
    """Use LLM to analyze a dataset referenced by the user."""
    
    system_prompt = (
        "You are an AI assistant helping users explore datasets. "
        "The user has asked about a dataset they previously searched for. "
        "Based on the available datasets, identify which one they are referring to and provide insights."
    )

    dataset_titles = [f"{i+1}. {data['title']}" for i, data in enumerate(previous_results)]
    
    query_prompt = f"User's question: {user_query}\nAvailable Datasets:\n{', '.join(dataset_titles)}\nProvide insights."

    response = ollama.chat(
        model="llama2:13b",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query_prompt}]
    )

    bot_response = response["message"]["content"]
    chat_history.append({"role": "assistant", "content": bot_response})
    return bot_response



def refine_search_with_llm(user_query):
    """LLM suggests improvements if search results were not ideal."""
    query_prompt = f"User's original query: {user_query}\nSuggest a better search query."

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query_prompt}]
    )

    return response["message"]["content"]


def chatbot_loop():
    """Fully conversational chatbot using FAISS for dataset search and Llama2-13B for dataset analysis."""
    
    print("🤖 Chatbot is running! Type 'exit' to quit.\n")

    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # 🔥 Generate chatbot response
        response = generate_chatbot_response(user_input)
        print(f"\n🤖 Chatbot:\n{response}")

        # ✅ Trim chat history to last 10 messages (for memory efficiency)
        if len(chat_history) > 10:
            chat_history.pop(0)


if __name__ == "__main__":
    chatbot_loop()
