import os
import json
import logging
import requests

# ✅ Setup
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(__file__)
EVAL_PATH = os.path.join(BASE_DIR, "../evaluation_dataset.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "keyword_results.json")

# ✅ Load dataset
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

# ✅ Manual keywords for each conversation
manual_keywords = {
    "Academic Conferences#1": "ai conferences",
    "Academic Conference#2": "ai",
    "Property Prices #1": "average property prices",
    "Property Prices #2": "property prices",
    "Birds#1": "ebird",
    "Birds#2": "birds",
    "Postcode Address Files #1": "UK Ordnance Survey Open Names",
    "Postcode Address Files #2": "streetnames",
    "Laws #1": "tenancy",
    "Laws #2": "tenancy",
    "Education #1": "university enrollment",
    "Education #2": "university gender",
    "Homelessness #1": "homelessness",
    "Homelessness #2": "homelessness trends",
    "Physics #1": "fusion",
    "Physics #2": "fusion",
    "Food #1": "food budget",
    "Food #2": "households food",
    "Tree Canopy Cover #1": "tree canopy",
    "Tree Canopy Cover #2": "tree canopy cover",
    "Sport and Activity #1": "sports",
    "Sport and Activity #2": "sports",
    "Census #1": "census",
    "Census #2": "census data",
    "Travel #1": "travel infrastructure",
    "Travel #2": "active travel infrastructure",
    "Legal Aid #1": "legal aid",
    "Legal Aid #2": "legal aid",
    "Prisons #1": "prisons",
    "Prisons #2": "prisons",
    "Carers #1": "carers",
    "Carers #2": "carers",
    "Universal Credit/Medical #1": "grocery purchase",
    "Universal Credit/Medical #2": "grocery purchas",
    "Rainfall #1": "rainfall",
    "Rainfall #2": "active rainfall tracking",
    "Healthcare #1": "diabetes",
    "Healthcare #2": "healthcare",
    "Dog Bites #1": "dog bites",
    "Dog Bites #2": "dog bites",
    "Sea Temperatures #1": "sea temperature",
    "Sea Temperatures #2": "sea temperature",
    "Population Trends #1": "population trends",
    "Population Trends #2": "population",
    "Traffic #1": "traffic",
    "Traffic #2": "traffic",
    "Transport Usage #1": "transport accessibility",
    "Transport Usage #2": "flight traffic",
    "Violence #1": "child violence",
    "Violence #2": "child violence",
    "Weather #1": "weather",
    "Weather #2": "weather",
    "Sports #1": "sports participation",
    "Sports #2": "physical exercise",
    "Air Quality #1": "air pollution",
    "Air Quality #2": "air pollution trends",
    "Water Quality #1": "water quality",
    "Water Quality #2": "water quality climate change",
    "Cost of Living #1": "inflation rates",
    "Cost of Living #2": "cost of living",
    "Energy Consumption #1": "energy consumption",
    "Energy Comsumption #2": "renewables",
    "Migration #1": "immigration trends",
    "Migration #2": "emigration"
}

# ✅ API base URL
API_URL = "https://data.london.gov.uk/api/3/action/package_search"

def query_london_datastore(keywords):
    """Search London Datastore using keywords."""
    try:
        resp = requests.get(
            API_URL,
            params={"q": keywords, "rows": 5},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10
        )

        print(f"➡️ API Status: {resp.status_code}")
        print(f"📦 API Response: {resp.json()}")

        data = resp.json()

        if not data.get("success"):
            logging.warning(f"❌ API query failed: {data.get('error')}")
            return []

        # ✅ Fix here
        results = data["result"]["result"]
        return [res["title"] for res in results]

    except Exception as e:
        logging.warning(f"⚠️ Failed to fetch results for '{keywords}': {e}")
        return []


def main():
    baseline_results = []

    for convo in evaluation_data:
        convo_id = convo["conversation_id"]
        keyword = manual_keywords.get(convo_id)

        if not keyword:
            logging.info(f"⏭️ Skipping {convo_id}: No manual keyword")
            continue

        convo_output = {"conversation_id": convo_id, "turns": []}

        for turn in convo["turns"]:
            user_query = turn["user"]
            eval_type = turn["eval"]["type"].lower()
            ground_truth = turn["eval"].get("ground_truth_ld", [])

            # ✅ Check if this turn is a dataset-relevant type
            if eval_type in ["described dataset", "dataset request", "implied dataset"]:
                # ✅ Skip if there's no ground truth
                if not ground_truth:
                    convo_output["turns"].append({
                        "user": user_query,
                        "eval_type": eval_type,
                        "result": "skipped (no ground truth)"
                    })
                    continue

                # ✅ Run search with manual keyword
                results = query_london_datastore(keyword)
                convo_output["turns"].append({
                    "user": user_query,
                    "eval_type": eval_type,
                    "keywords": keyword,
                    "top_results": results
                })

                logging.info(f"🔍 [{convo_id}] '{user_query}' → {keyword} → {results}")

            else:
                convo_output["turns"].append({
                    "user": user_query,
                    "eval_type": eval_type,
                    "result": "skipped"
                })

        baseline_results.append(convo_output)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ Baseline results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
