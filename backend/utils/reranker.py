from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Upgraded Reranker Model
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# 💡 Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def rerank(query, results):
    """Batch reranking using a stronger cross-encoder model."""
    rerank_scores = []
    query_lower = query.lower()

    # ✅ Identify format preference
    format_preference = None
    for fmt in ["spreadsheet", "csv", "pdf", "zip"]:
        if fmt in query_lower:
            format_preference = fmt
            break

    # ✅ Precompute input pairs and boost factors
    text_pairs = []
    format_boosts = []
    year_boosts = []

    for res in results:
        format_boost = 1.5 if format_preference and format_preference in res.get("format", []) else 1.0
        format_boosts.append(format_boost)

        temporal_from = res["temporal_coverage"].get("from")
        year_boost = 1.0
        if temporal_from and isinstance(temporal_from, str) and temporal_from[:4].isdigit():
            year_boost = 1.0 + (int(temporal_from[:4]) - 2000) / 50
        year_boosts.append(year_boost)

        text_pairs.append((
            query,
            f"Title: {res['title']}. Summary: {res['summary']}. "
            f"Publisher: {res['publisher']}. Tags: {', '.join(res['tags'])}. "
            f"Temporal: {temporal_from} - {res['temporal_coverage']['to']}. "
            f"Geospatial: {res['geospatial_coverage'].get('bounding_box', 'Unknown')}. "
            f"Format: {', '.join(res['format'])}"
        ))

    # ✅ Tokenize batch
    inputs = tokenizer([q for q, d in text_pairs],
                       [d for q, d in text_pairs],
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=512).to(device)

    # ✅ Run inference
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze()
        if len(logits.shape) == 0:
            scores = [logits.item()]
        else:
            scores = logits.cpu().tolist()

    # ✅ Apply boosts
    rerank_scores = [s * f * y for s, f, y in zip(scores, format_boosts, year_boosts)]

    # ✅ Attach scores
    for i, res in enumerate(results):
        res["rerank_score"] = rerank_scores[i]

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
