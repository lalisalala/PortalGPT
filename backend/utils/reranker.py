from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ Load Small Reranker Model (BAAI/bge-reranker-base)
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def rerank(query, results):
    """Optimized batch processing for reranking."""
    rerank_scores = []
    query_lower = query.lower()

    # ✅ Identify format preference in query
    format_preference = None
    for fmt in ["spreadsheet", "csv", "pdf", "zip"]:
        if fmt in query_lower:
            format_preference = fmt
            break

    # ✅ Precompute reranker inputs
    text_pairs = []
    format_boosts = []
    year_boosts = []

    for res in results:
        # ✅ Format preference boost
        format_boost = 1.5 if format_preference and format_preference in res.get("format", []) else 1.0
        format_boosts.append(format_boost)

        # ✅ Recency boost
        temporal_from = res["temporal_coverage"].get("from")
        year_boost = 1.0
        if temporal_from and isinstance(temporal_from, str) and temporal_from[:4].isdigit():
            year_boost = 1.0 + (int(temporal_from[:4]) - 2000) / 50  # Normalize from year 2000 onwards
        year_boosts.append(year_boost)

        # ✅ Concatenate metadata into reranker input
        text_pairs.append(
            f"Query: {query} [SEP] "
            f"Title: {res['title']} [SEP] "
            f"Summary: {res['summary']} [SEP] "
            f"Publisher: {res['publisher']} [SEP] "
            f"Tags: {', '.join(res['tags'])} [SEP] "
            f"Temporal Coverage: {temporal_from} - {res['temporal_coverage']['to']} [SEP] "
            f"Geospatial Coverage: {res['geospatial_coverage'].get('bounding_box', 'Unknown')} [SEP] "
            f"Format: {', '.join(res['format'])}"
        )

    # ✅ Tokenize entire batch at once (faster than looping)
    inputs = tokenizer(text_pairs, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)

    # ✅ Run model inference in one batch (GPU parallelization)
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze().cpu().tolist()

    # ✅ Apply format & recency boosts
    rerank_scores = [s * f * y for s, f, y in zip(scores, format_boosts, year_boosts)]

    # ✅ Attach scores & sort
    for i, res in enumerate(results):
        res["rerank_score"] = rerank_scores[i]

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
