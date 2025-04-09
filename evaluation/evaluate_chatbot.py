
import json
import os
import logging
from difflib import SequenceMatcher
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
# Load semantic similarity model once
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Load evaluation dataset
EVAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "evaluation_dataset.json")

with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

# ✅ Import the chatbot function
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm.answer_generator import generate_response, generate_user_id
from backend.utils.retrieval import search_faiss
# ✅ Matching utilities
def fuzzy_match(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def score_dataset_retrieval(predicted: List[str], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
    true_titles = [gt["dataset_title"] for gt in ground_truth]
    relevance = {gt["dataset_title"]: gt["relevance"] for gt in ground_truth}
    predicted_set = set(predicted)

    hits = [title for title in predicted if title in true_titles]
    precision = len(hits) / len(predicted) if predicted else 0
    recall = len(hits) / len(true_titles) if true_titles else 0
    avg_relevance = sum(relevance.get(title, 0) for title in hits) / len(hits) if hits else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "avg_relevance": round(avg_relevance, 2)
    }


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())

def fuzzy_match_titles(gt_titles, predicted_titles, threshold=0.7):
    matched = set()
    for gt in gt_titles:
        norm_gt = normalize(gt)
        for pred in predicted_titles:
            norm_pred = normalize(pred)
            if fuzzy_match(norm_gt, norm_pred) >= threshold:
                matched.add(gt)
                break
    return matched
def ndcg_score(relevance_scores, k=5):
    """Compute NDCG@k given a list of relevance scores."""
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        rel = relevance_scores[i]
        dcg += (2**rel - 1) / np.log2(i + 2)

    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_scores))):
        rel = ideal_scores[i]
        idcg += (2**rel - 1) / np.log2(i + 2)

    return round(dcg / idcg, 3) if idcg > 0 else 0.0



def evaluate_turn(user_query, ground_truth, eval_type, user_id):
# ✅ Direct FAISS retrieval evaluation (pre-LLM)
    if eval_type.lower() in ["described dataset", "dataset request", "implied dataset"]:
        gt_titles = [gt["dataset_title"] for gt in ground_truth]
        relevance_map = {gt["dataset_title"]: gt["relevance"] for gt in ground_truth}

        retrieved_datasets = search_faiss(user_query, k=10)
        predicted_titles = [ds["title"] for ds in retrieved_datasets[:5]]
        rel_scores = [relevance_map.get(title, 0) for title in predicted_titles]
        ndcg = ndcg_score(rel_scores, k=5)

        matched_titles = fuzzy_match_titles(gt_titles, predicted_titles, threshold=0.7)

        precision = len(matched_titles) / len(predicted_titles) if predicted_titles else 0
        recall = len(matched_titles) / len(gt_titles) if gt_titles else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        avg_relevance = (
            sum(relevance_map.get(title, 0) for title in matched_titles) / len(matched_titles)
            if matched_titles else 0
        )

        return {
            "result": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "avg_relevance": round(avg_relevance, 2),
                "ndcg": ndcg
            },
            "retrieved_titles": predicted_titles
        }


    # ✅ LLM-based response for metadata, links, etc.
    response = generate_response(user_query, user_id)

    if eval_type.lower() in ["link request", "source request"]:
        def normalize_url(url):
            return url.lower().strip().rstrip("/").replace("http://", "https://").replace("www.", "")

        predicted_links = [normalize_url(word) for word in response.split() if "http" in word]
        gt_links = [normalize_url(link) for link in ground_truth if isinstance(link, str)]

        correct = 0
        for pred in predicted_links:
            for gt in gt_links:
                if pred in gt or gt in pred:
                    correct += 1
                    break

        precision = correct / len(predicted_links) if predicted_links else 0
        recall = correct / len(gt_links) if gt_links else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {
            "result": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3)
            },
            "bot_response": response
        }

    if eval_type.lower() in ["additional dataset information"]:
        predicted_facts = [line.strip() for line in response.split("\n") if line.strip()]
        correct = 0
        for gt in ground_truth:
            gt_embedding = sbert_model.encode(gt, convert_to_tensor=True)
            for pred in predicted_facts:
                pred_embedding = sbert_model.encode(pred, convert_to_tensor=True)
                if util.cos_sim(gt_embedding, pred_embedding).item() > 0.8:
                    correct += 1
                    break

        return {
            "result": {
                "correctness": round(correct / len(ground_truth), 3) if ground_truth else 0
            },
            "bot_response": response
        }

    return {"result": {"note": f"⚠️ Unsupported eval type: {eval_type}"}, "bot_response": response}

# ✅ Run Evaluation
# ✅ Run Evaluation (updated to skip turns with empty ground truth)
def main():
    results = []
    flat_rows = []  # For CSV export
    skipped_turns = 0
    total_turns = 0
    initial_search_count = 0  # 🔥 Count how often FAISS is triggered

    for convo in evaluation_data:
        user_id = generate_user_id()
        convo_id = convo["conversation_id"]
        logging.info(f"📂 Evaluating conversation: {convo_id}")
        convo_results = {"conversation_id": convo_id, "turns": []}

        for turn_idx, turn in enumerate(convo["turns"]):
            total_turns += 1
            query = turn["user"]
            eval_type = turn["eval"]["type"]
            ground_truth = turn["eval"]["ground_truth_ld"]

            if not ground_truth:
                logging.info(f"⏭️ Skipping turn (no ground truth): \"{query}\"")
                skipped_turns += 1
                convo_results["turns"].append({
                    "user": query,
                    "type": eval_type,
                    "result": "skipped (no ground truth)"
                })
                continue

            # 🔥 Count only dataset-search-like intents
            if eval_type.lower() in ["described dataset", "dataset request", "implied dataset"]:
                initial_search_count += 1

            eval_output = evaluate_turn(query, ground_truth, eval_type, user_id)
            result = eval_output["result"]

            bot_response = (
                eval_output.get("retrieved_titles")
                if "retrieved_titles" in eval_output
                else eval_output.get("bot_response", "N/A")
            )

            convo_results["turns"].append({
                "user": query,
                "type": eval_type,
                "ground_truth_ld": ground_truth,
                "bot_response": bot_response,
                "result": result
            })

            flat_row = {
                "conversation_id": convo_id,
                "turn_index": turn_idx,
                "user_query": query,
                "eval_type": eval_type,
                "ground_truth": json.dumps(ground_truth),
                "bot_response": json.dumps(bot_response) if isinstance(bot_response, list) else bot_response
            }
            flat_row.update(result)
            flat_rows.append(flat_row)

        results.append(convo_results)

    answered_turns = total_turns - skipped_turns

    # ✅ Summary log
    logging.info("📊 Evaluation Summary")
    logging.info(f"🔹 Total Conversations: {len(evaluation_data)}")
    logging.info(f"🔹 Total Turns: {total_turns}")
    logging.info(f"🔹 Answered Turns: {answered_turns}")
    logging.info(f"🔹 Skipped Turns (no ground truth): {skipped_turns}")
    logging.info(f"🧠 Initial Dataset Searches Triggered: {initial_search_count}")

    # ✅ Save JSON
    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"✅ JSON saved to: {output_path}")

    # ✅ Save CSV
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), "evaluation_results_flat.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)
    logging.info(f"📄 CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
