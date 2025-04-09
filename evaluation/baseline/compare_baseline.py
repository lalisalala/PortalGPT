import os
import json
from collections import defaultdict
import numpy as np

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
EVAL_PATH = os.path.join(BASE_DIR, "../evaluation_dataset.json")

# All baseline files you want to evaluate
BASELINE_FILES = {
    "keyword": os.path.join(BASE_DIR, "keyword_results.json"),       # ✅ renamed
    "chatgpt": os.path.join(BASE_DIR, "chatgpt_results.json")        # ✅ second baseline
}

OUTPUT_DIR = BASE_DIR  # where evaluation results will be saved

# --- Load ground truth ---
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    ground_truth_data = {c["conversation_id"]: c for c in json.load(f)}

# --- Utility ---
def relevance_score(title, truth):
    for item in truth:
        if title.lower().strip() == item["dataset_title"].lower().strip():
            return item["relevance"]
    return 0
def ndcg_score(relevance_scores, k=5):
    """Compute NDCG@k given a list of relevance scores."""
    dcg = 0.0
    for i in range(min(k, len(relevance_scores))):
        rel = relevance_scores[i]
        dcg += (2**rel - 1) / (np.log2(i + 2))

    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_scores))):
        rel = ideal_scores[i]
        idcg += (2**rel - 1) / (np.log2(i + 2))

    return round(dcg / idcg, 3) if idcg > 0 else 0.0
def evaluate_turn(predictions, ground_truth):
    if not ground_truth:
        return "skipped (no ground truth)"

    pred_set = set(predictions)
    gold_titles = set(item["dataset_title"] for item in ground_truth)

    true_positives = pred_set & gold_titles
    precision = len(true_positives) / len(pred_set) if pred_set else 0
    recall = len(true_positives) / len(gold_titles) if gold_titles else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    avg_relevance = 0
    if true_positives:
        avg_relevance = sum(relevance_score(t, ground_truth) for t in true_positives) / len(true_positives)

    rel_scores = [relevance_score(t, ground_truth) for t in predictions]
    ndcg = ndcg_score(rel_scores, k=5)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "avg_relevance": round(avg_relevance, 2),
        "ndcg": ndcg
    }

# --- Evaluate All Baselines ---
for label, path in BASELINE_FILES.items():
    with open(path, "r", encoding="utf-8") as f:
        baseline_data = {c["conversation_id"]: c for c in json.load(f)}

    baseline_eval_results = []

    for convo_id, baseline in baseline_data.items():
        eval_convo = ground_truth_data.get(convo_id)
        if not eval_convo:
            continue

        convo_output = {
            "conversation_id": convo_id,
            "turns": []
        }

        for i, turn in enumerate(baseline["turns"]):
            eval_type = turn["eval_type"]
            user_query = turn["user"]
            top_results = turn.get("top_results", [])
            gt_turn = eval_convo["turns"][i]
            ground_truth = gt_turn.get("eval", {}).get("ground_truth_ld", [])

            if not ground_truth:
                result = "skipped (no ground truth)"
            elif eval_type.lower() in ["described dataset", "dataset request", "implied dataset"]:
                result = evaluate_turn(top_results, ground_truth)
            else:
                result = "skipped"

            convo_output["turns"].append({
                "user": user_query,
                "eval_type": eval_type,
                "top_results": top_results,
                "result": result
            })

        baseline_eval_results.append(convo_output)

    output_path = os.path.join(OUTPUT_DIR, f"{label}_baseline.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline_eval_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Evaluated and saved: {output_path}")
