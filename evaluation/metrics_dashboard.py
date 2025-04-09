import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib import cm

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10
})

colors = cm.get_cmap("Set2").colors  # cleaner and softer

# --- File Paths ---
BASE_DIR = os.path.dirname(__file__)
CHATBOT_PATH = os.path.join(BASE_DIR, "evaluation_results.json")
KEYWORD_PATH = os.path.join(BASE_DIR, "baseline/keyword_baseline.json")
CHATGPT_PATH = os.path.join(BASE_DIR, "baseline/chatgpt_baseline.json")
PLOT_PATH = os.path.join(BASE_DIR, "evaluation_metrics_all_models.png")

# --- Load & Aggregate ---
def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics_by_type = defaultdict(list)
    for convo in data:
        for turn in convo["turns"]:
            eval_type = turn.get("eval_type", turn.get("type", "")).strip().lower()
            result = turn.get("result", {})
            if isinstance(result, dict):
                metrics_by_type[eval_type].append(result)

    summary = {}
    for eval_type, results in metrics_by_type.items():
        agg = defaultdict(list)
        for r in results:
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    agg[k].append(v)
        summary[eval_type] = {k: round(np.mean(v), 3) for k, v in agg.items()}
    return summary

chatbot = load_results(CHATBOT_PATH)
keyword = load_results(KEYWORD_PATH)
chatgpt = load_results(CHATGPT_PATH)

# --- Log Raw Metric Summaries ---
def print_summary(label, summary_dict):
    print(f"\n📊 {label} Evaluation Summary:")
    for eval_type, metrics in summary_dict.items():
        print(f"  🔹 {eval_type.title()}:")
        for metric_name, value in metrics.items():
            print(f"     {metric_name}: {value:.3f}")

print_summary("Chatbot", chatbot)
print_summary("Keyword Baseline", keyword)
print_summary("ChatGPT Baseline", chatgpt)

# --- Prepare plotting ---
eval_types = sorted(set(chatbot) | set(keyword) | set(chatgpt))
metrics = ["precision", "recall", "f1", "avg_relevance", "ndcg"]
models = ["Chatbot", "Keyword", "ChatGPT"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

bar_width = 0.25
x = np.arange(len(eval_types))

fig, axs = plt.subplots(len(metrics), 1, figsize=(15, 10), sharex=True)

for i, metric in enumerate(metrics):
    ax = axs[i]
    ax.set_title(f"{metric.capitalize()} by Model", fontsize=12, fontweight="bold")
    # Values per model
    chatbot_vals = [chatbot.get(et, {}).get(metric, 0) for et in eval_types]
    keyword_vals = [keyword.get(et, {}).get(metric, 0) for et in eval_types]
    chatgpt_vals = [chatgpt.get(et, {}).get(metric, 0) for et in eval_types]

    # Offsets
    bars1 = ax.bar(x - bar_width, chatbot_vals, width=bar_width, label="Chatbot", color=colors[0])
    bars2 = ax.bar(x, keyword_vals, width=bar_width, label="Keyword", color=colors[1])
    bars3 = ax.bar(x + bar_width, chatgpt_vals, width=bar_width, label="ChatGPT", color=colors[2])
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    if metric == "avg_relevance":
        ax.set_ylim(0, 4.2)
    elif metric == "ndcg":
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, 1.05)


    # Add legend only to the first plot
    if i == 0:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

axs[-1].set_xticks(x)
axs[-1].set_xticklabels(eval_types, rotation=45, ha="right")
axs[-1].set_xlabel("Evaluation Type", fontsize=12)

plt.suptitle("📊 Evaluation Comparison Across Models", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(PLOT_PATH)
plt.close()

print(f"📈 Saved metric comparison plot to {PLOT_PATH}")
