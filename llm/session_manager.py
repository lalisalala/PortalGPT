import uuid
import logging

# ✅ Store user sessions
user_sessions = {}

def generate_user_id():
    """Generates a unique user ID."""
    return str(uuid.uuid4())

def get_user_session(user_id):
    """Retrieves user session, ensuring it is initialized."""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "results": [],
            "shown_count": 0,
            "has_searched": False  # ✅ Initialize the search flag
        }
    return user_sessions[user_id]


def update_user_history(user_id, query):
    """Stores user queries in session history."""
    session = get_user_session(user_id)
    session["history"].append(query)  # ✅ Append query to session history
    logging.info(f"📝 Added query to session history: {query}")


def store_faiss_results(user_id, results):
    """Stores FAISS search results and ensures pagination is not reset unnecessarily."""
    session = get_user_session(user_id)

    logging.info(f"📌 Before storing: shown_count = {session.get('shown_count', 0)}")

    # ✅ Only reset results if it's a **completely new** search, not a follow-up request
    if session["results"] != results:
        session["results"] = results
        session["shown_count"] = 0  # ✅ Reset count only for a **brand new search**
    else:
        logging.info(f"🔄 Keeping shown_count at {session['shown_count']} since results are unchanged.")

    logging.info(f"📌 After storing: shown_count = {session['shown_count']} for user {user_id}")
    logging.info(f"✅ FAISS results stored. Total results: {len(results)}")



def get_next_faiss_results(user_id, batch_size=3):
    """Retrieves the next batch of FAISS results sequentially, ensuring proper pagination."""
    session = get_user_session(user_id)  # ✅ Ensure session exists
    all_results = session["results"]
    shown_count = session["shown_count"]

    logging.info(f"🟡 Retrieving more datasets for user {user_id}")
    logging.info(f"📊 Total FAISS results: {len(all_results)} | Already shown: {shown_count}")

    if shown_count >= len(all_results):
        logging.warning(f"⚠️ No more datasets available for user {user_id}")
        return []

    # ✅ First, increment `shown_count` before fetching results
    new_shown_count = min(shown_count + batch_size, len(all_results))

    # ✅ Fetch the correct next batch
    next_batch = all_results[shown_count:new_shown_count]

    # ✅ Update session before returning results
    session["shown_count"] = new_shown_count

    logging.info(f"✅ Fetching datasets {shown_count + 1} to {new_shown_count} for user {user_id}")
    logging.info(f"🔍 Next batch dataset titles: {[r['title'] for r in next_batch]}")  # ✅ Log dataset titles

    return next_batch

def get_last_shown_datasets(user_id, count=1):
    """Returns the most recently shown `count` datasets from the user's session."""
    session = get_user_session(user_id)
    all_results = session.get("results", [])
    shown_count = session.get("shown_count", 0)

    # ✅ Prevent negative slicing if nothing has been shown yet
    if shown_count == 0 or not all_results:
        return []

    start = max(0, shown_count - count)
    return all_results[start:shown_count]
