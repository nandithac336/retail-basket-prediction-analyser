"""Core implementation for the Retail Basket Prediction Analyzer.

The project is deliberately laptop-friendly: a compact neural next-item language
model is pretrained on verbalized basket sequences with NumPy. This avoids
platform-specific GPU requirements while keeping the full training/evaluation loop.
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

CATALOG = {
    "Bakery": ["Bread", "Bagel", "Croissant", "Muffin", "Crackers", "Tortilla"],
    "Dairy": ["Milk", "Butter", "Cheese", "Yogurt", "Cream", "Eggs"],
    "Produce": ["Apples", "Bananas", "Tomatoes", "Onions", "Potatoes", "Lettuce"],
    "Pantry": ["Rice", "Pasta", "Tomato_Sauce", "Olive_Oil", "Flour", "Cereal"],
    "Snacks": ["Chips", "Soda", "Cookies", "Chocolate", "Popcorn", "Juice"],
    "Household": ["Soap", "Shampoo", "Detergent", "Tissue", "Toothpaste", "Dish_Soap"],
    "Protein": ["Chicken", "Tuna", "Beans", "Lentils", "Tofu", "Sausage"],
}
ALL_ITEMS = [item for values in CATALOG.values() for item in values]

# 32 planted itemsets: each is a synthetic shopping mission / co-purchase pattern.
PLANTED_ITEMSETS = [
    ["Bread", "Butter", "Milk"], ["Bread", "Cheese", "Eggs"],
    ["Bagel", "Cream", "Juice"], ["Croissant", "Butter", "Coffee"],
    ["Pasta", "Tomato_Sauce", "Cheese"], ["Rice", "Beans", "Olive_Oil"],
    ["Chicken", "Rice", "Olive_Oil"], ["Tuna", "Bread", "Lettuce"],
    ["Chips", "Soda", "Salsa"], ["Cookies", "Milk", "Chocolate"],
    ["Popcorn", "Soda", "Chocolate"], ["Cereal", "Milk", "Bananas"],
    ["Yogurt", "Bananas", "Cereal"], ["Apples", "Juice", "Yogurt"],
    ["Tomatoes", "Onions", "Olive_Oil"], ["Potatoes", "Onions", "Sausage"],
    ["Lettuce", "Tomatoes", "Cheese"], ["Flour", "Eggs", "Butter"],
    ["Soap", "Shampoo", "Tissue"], ["Detergent", "Dish_Soap", "Tissue"],
    ["Toothpaste", "Soap", "Shampoo"], ["Crackers", "Cheese", "Tuna"],
    ["Tortilla", "Beans", "Cheese"], ["Pasta", "Olive_Oil", "Tomatoes"],
    ["Lentils", "Rice", "Onions"], ["Tofu", "Rice", "Soy_Sauce"],
    ["Muffin", "Juice", "Apples"], ["Sausage", "Eggs", "Bread"],
    ["Cream", "Pasta", "Cheese"], ["Chicken", "Potatoes", "Carrots"],
    ["Chocolate", "Cookies", "Milk"], ["Popcorn", "Juice", "Chips"],
]
# Add a few products used by planted bundles but not in the main catalogue.
for _extra in ("Coffee", "Salsa", "Soy_Sauce", "Carrots"):
    if _extra not in ALL_ITEMS:
        ALL_ITEMS.append(_extra)


def milestone_1(n_baskets: int = 5000, seed: int = 42, output_dir: str | Path | None = None):
    """Generate reproducible synthetic baskets with planted itemsets and background noise."""
    if n_baskets < 100:
        raise ValueError("n_baskets must be at least 100")
    rng = random.Random(seed)
    out = Path(output_dir) if output_dir else DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    # Popularity-weighted background purchases.
    popularity = {item: rng.uniform(0.4, 1.8) for item in ALL_ITEMS}
    total = sum(popularity.values())
    probs = [popularity[item] / total for item in ALL_ITEMS]
    baskets = []
    for basket_id in range(1, n_baskets + 1):
        basket = []
        # 72% of baskets contain one or more planted missions.
        if rng.random() < 0.72:
            bundle = rng.choice(PLANTED_ITEMSETS)
            basket.extend(bundle)
            # Some planted bundles have realistic missing-item noise.
            if rng.random() < 0.12 and len(basket) > 2:
                basket.pop(rng.randrange(len(basket)))
            # 12% chance of a second planted bundle, adding cross-category variety.
            if rng.random() < 0.12:
                second = rng.choice(PLANTED_ITEMSETS)
                basket.extend(second)
        # Add 1–4 background products.
        extra_count = rng.randint(1, 4)
        basket.extend(rng.choices(ALL_ITEMS, weights=probs, k=extra_count))
        basket = list(dict.fromkeys(basket))
        rng.shuffle(basket)
        baskets.append(basket)

    csv_path = out / "baskets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["basket_id", "item_position", "item"])
        for basket_id, basket in enumerate(baskets, 1):
            for position, item in enumerate(basket, 1):
                writer.writerow([basket_id, position, item])

    truth_path = out / "planted_itemsets.json"
    truth_path.write_text(json.dumps(PLANTED_ITEMSETS, indent=2), encoding="utf-8")
    return {"baskets": baskets, "csv_path": str(csv_path), "truth_path": str(truth_path),
            "n_baskets": len(baskets), "n_itemsets": len(PLANTED_ITEMSETS),
            "n_products": len(ALL_ITEMS)}


def load_baskets(path: str | Path | None = None):
    """Load generated baskets from the item-per-row CSV."""
    path = Path(path) if path else DATA_DIR / "baskets.csv"
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            grouped[int(row["basket_id"])].append((int(row["item_position"]), row["item"]))
    return [[item for _, item in sorted(rows)] for _, rows in sorted(grouped.items())]


def _split_baskets(baskets, test_ratio=0.2, seed=13):
    indices = list(range(len(baskets)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * (1 - test_ratio))
    return [baskets[i] for i in indices[:cut]], [baskets[i] for i in indices[cut:]]


class NGramModel:
    """Frequency-based next-item model with bigram/trigram context backoff."""
    def __init__(self, max_order=3):
        self.max_order = max_order
        self.counts = {n: defaultdict(Counter) for n in range(1, max_order)}
        self.unigram = Counter()

    def fit(self, baskets):
        for basket in baskets:
            for i, target in enumerate(basket):
                self.unigram[target] += 1
                for n in range(1, self.max_order):
                    if i >= n:
                        self.counts[n][tuple(basket[i-n:i])][target] += 1
        return self

    def predict_scores(self, context):
        for n in range(min(self.max_order - 1, len(context)), 0, -1):
            counts = self.counts[n].get(tuple(context[-n:]))
            if counts:
                return dict(counts)
        return dict(self.unigram)

    def predict(self, context, k=5):
        scores = self.predict_scores(context)
        return [item for item, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]]


def milestone_2(baskets=None, max_order=3):
    """Train the n-gram baseline on the supplied baskets."""
    if baskets is None:
        baskets = load_baskets()
    train, test = _split_baskets(baskets)
    model = NGramModel(max_order=max_order).fit(train)
    return {"model": model, "train_baskets": train, "test_baskets": test}


class TinyNeuralLanguageModel:
    """A compact neural language model: item embeddings -> tanh -> softmax.

    It learns to predict the next item from the preceding three items. The
    weights are trained on basket sequences (continued domain pretraining idea).
    """
    def __init__(self, vocab_size, context_size=3, embedding_dim=16, hidden_dim=48, seed=7):
        rng = np.random.default_rng(seed)
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.E = rng.normal(0, 0.08, (vocab_size, embedding_dim)).astype(np.float32)
        self.W1 = rng.normal(0, 0.07, (context_size * embedding_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, 0.07, (hidden_dim, vocab_size)).astype(np.float32)
        self.b2 = np.zeros(vocab_size, dtype=np.float32)

    def _forward(self, contexts):
        emb = self.E[contexts]
        flat = emb.reshape(len(contexts), -1)
        hidden = np.tanh(flat @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return emb, flat, hidden, probs

    def fit(self, examples, epochs=80, learning_rate=0.06, batch_size=128, seed=9):
        X = np.asarray([x for x, _ in examples], dtype=np.int64)
        y = np.asarray([y for _, y in examples], dtype=np.int64)
        rng = np.random.default_rng(seed)
        history = []
        for epoch in range(epochs):
            order = rng.permutation(len(X))
            total_loss = 0.0
            for start in range(0, len(X), batch_size):
                ix = order[start:start + batch_size]
                xb, yb = X[ix], y[ix]
                emb, flat, hidden, probs = self._forward(xb)
                n = len(ix)
                loss = -np.log(probs[np.arange(n), yb] + 1e-9).mean()
                total_loss += float(loss) * n

                dlogits = probs.copy()
                dlogits[np.arange(n), yb] -= 1.0
                dlogits /= n
                dW2 = hidden.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dh = dlogits @ self.W2.T
                dz = dh * (1.0 - hidden * hidden)
                dW1 = flat.T @ dz
                db1 = dz.sum(axis=0)
                dflat = dz @ self.W1.T
                demb = dflat.reshape(n, self.context_size, self.embedding_dim)
                dE = np.zeros_like(self.E)
                for pos in range(self.context_size):
                    np.add.at(dE, xb[:, pos], demb[:, pos, :])

                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.E -= learning_rate * dE
            history.append(total_loss / max(1, len(X)))
        return history

    def predict_scores(self, context):
        context = list(context)[-self.context_size:]
        context = [0] * (self.context_size - len(context)) + context
        _, _, _, probs = self._forward(np.asarray([context], dtype=np.int64))
        return probs[0]

    def save(self, path):
        np.savez(path, E=self.E, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 context_size=self.context_size, embedding_dim=self.embedding_dim)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(data["E"].shape[0], int(data["context_size"]),
                    int(data["embedding_dim"]), data["W1"].shape[1])
        for name in ("E", "W1", "b1", "W2", "b2"):
            setattr(model, name, data[name])
        return model


def milestone_3(baskets=None, epochs=80, output_dir: str | Path | None = None):
    """Pretrain the compact neural language model on verbalized basket sequences."""
    if baskets is None:
        baskets = load_baskets()
    train, test = _split_baskets(baskets)
    vocabulary = sorted({item for basket in train for item in basket})
    item_to_id = {item: i + 1 for i, item in enumerate(vocabulary)}  # 0 = PAD
    id_to_item = {v: k for k, v in item_to_id.items()}
    context_size = 3
    examples = []
    for basket in train:
        ids = [item_to_id[x] for x in basket]
        for i in range(1, len(ids)):
            context = ids[max(0, i-context_size):i]
            context = [0] * (context_size-len(context)) + context
            examples.append((context, ids[i]))
    model = TinyNeuralLanguageModel(len(vocabulary) + 1, context_size=context_size)
    history = model.fit(examples, epochs=epochs)
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "basket_language_model.npz")
    (out / "vocabulary.json").write_text(json.dumps(item_to_id, indent=2), encoding="utf-8")
    (out / "training_loss.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"model": model, "vocabulary": item_to_id, "id_to_item": id_to_item,
            "train_baskets": train, "test_baskets": test, "loss_history": history,
            "training_examples": len(examples)}


def _rank_metrics(ranks):
    n = max(1, len(ranks))
    return {
        "mrr": float(sum(1.0 / r for r in ranks) / n),
        "hits_at_1": float(sum(r <= 1 for r in ranks) / n),
        "hits_at_3": float(sum(r <= 3 for r in ranks) / n),
        "hits_at_5": float(sum(r <= 5 for r in ranks) / n),
        "mean_rank": float(sum(ranks) / n),
    }


def _rank_from_scores(scores, target, items):
    ordered = sorted(items, key=lambda item: (-scores.get(item, 0.0), item))
    try:
        return ordered.index(target) + 1
    except ValueError:
        return len(ordered) + 1


def milestone_4(ngram_result=None, neural_result=None, baskets=None):
    """Evaluate both models on identical held-out next-item transitions."""
    if baskets is None:
        baskets = load_baskets()
    train, test = _split_baskets(baskets)
    if ngram_result is None:
        ngram_result = {"model": NGramModel().fit(train)}
    if neural_result is None:
        # For a standalone evaluation call, train the neural model on the same split.
        vocab = sorted({item for basket in train for item in basket})
        ids = {item: i + 1 for i, item in enumerate(vocab)}
        examples = []
        for basket in train:
            seq = [ids[x] for x in basket]
            for i in range(1, len(seq)):
                ctx = seq[max(0, i-3):i]
                examples.append(([0] * (3-len(ctx)) + ctx, seq[i]))
        lm = TinyNeuralLanguageModel(len(vocab)+1)
        history = lm.fit(examples, epochs=80)
        neural_result = {"model": lm, "vocabulary": ids, "loss_history": history}
    ng_model = ngram_result["model"]
    nn_model = neural_result["model"]
    vocab = neural_result.get("vocabulary") or {item: i+1 for i, item in enumerate(sorted({x for b in train for x in b}))}
    item_ids = vocab
    all_items = sorted(set(item_ids))
    ngram_ranks, neural_ranks = [], []
    for basket in test:
        seq = [x for x in basket if x in item_ids]
        for i in range(1, len(seq)):
            context_items = seq[max(0, i-3):i]
            target = seq[i]
            n_scores = ng_model.predict_scores(context_items)
            ngram_ranks.append(_rank_from_scores(n_scores, target, all_items))
            context_ids = [item_ids[x] for x in context_items]
            nn_scores_arr = nn_model.predict_scores(context_ids)
            nn_scores = {item: float(nn_scores_arr[item_ids[item]]) for item in all_items}
            neural_ranks.append(_rank_from_scores(nn_scores, target, all_items))
    return {"ngram": _rank_metrics(ngram_ranks), "neural_lm": _rank_metrics(neural_ranks),
            "n_test_predictions": len(ngram_ranks), "ngram_ranks": ngram_ranks,
            "neural_ranks": neural_ranks}


def milestone_5(metrics, loss_history=None, output_dir: str | Path | None = None):
    """Save a metrics report and model-comparison visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in metrics.items() if not k.endswith("_ranks")}
    (out / "metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")

    names = ["N-gram", "Neural LM"]
    mrr = [metrics["ngram"]["mrr"], metrics["neural_lm"]["mrr"]]
    h3 = [metrics["ngram"]["hits_at_3"], metrics["neural_lm"]["hits_at_3"]]
    x = np.arange(2)
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-width/2, mrr, width, label="MRR")
    ax.bar(x+width/2, h3, width, label="Hits@3")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Next-item prediction: model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=160)
    plt.close(fig)

    if loss_history:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(range(1, len(loss_history)+1), loss_history, marker="o")
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Neural language model pretraining loss")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "training_loss.png", dpi=160)
        plt.close(fig)

    report = [
        "# Model comparison report", "",
        f"Test predictions evaluated: **{metrics['n_test_predictions']}**", "",
        "| Metric | N-gram | Neural LM |",
        "|---|---:|---:|",
    ]
    for key, label in [("mrr", "Mean Reciprocal Rank"), ("hits_at_1", "Hits@1"),
                       ("hits_at_3", "Hits@3"), ("hits_at_5", "Hits@5"),
                       ("mean_rank", "Mean Rank")]:
        report.append(f"| {label} | {metrics['ngram'][key]:.4f} | {metrics['neural_lm'][key]:.4f} |")
    report += ["", "Higher MRR and Hits@K are better; lower Mean Rank is better.",
               "The dataset is synthetic and contains planted co-purchase itemsets."]
    (out / "comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"metrics_path": str(out / "metrics.json"),
            "chart_path": str(out / "model_comparison.png"),
            "report_path": str(out / "comparison_report.md")}


def run_pipeline(n_baskets=5000, epochs=80):
    """Run all five milestones in order and return their outputs."""
    generated = milestone_1(n_baskets=n_baskets)
    baskets = generated["baskets"]
    ngram_result = milestone_2(baskets)
    neural_result = milestone_3(baskets, epochs=epochs)
    metrics = milestone_4(ngram_result, neural_result, baskets)
    artifacts = milestone_5(metrics, neural_result["loss_history"])
    return {"generated": generated, "ngram": ngram_result, "neural": neural_result,
            "metrics": metrics, "artifacts": artifacts}


def predict_next(items, model_path=None, vocab_path=None, top_k=5):
    """Load the trained neural model and predict next products for a comma-separated item list."""
    model_path = Path(model_path) if model_path else OUTPUT_DIR / "basket_language_model.npz"
    vocab_path = Path(vocab_path) if vocab_path else OUTPUT_DIR / "vocabulary.json"
    if not model_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Model not found. Run `python -m app.cli run` first.")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    model = TinyNeuralLanguageModel.load(model_path)
    ids = [vocab.get(item.strip().replace(" ", "_"), 0) for item in items if item.strip()]
    scores = model.predict_scores(ids)
    reverse = {int(v): k for k, v in vocab.items()}
    ranked = np.argsort(scores)[::-1]
    results = []
    for idx in ranked:
        if int(idx) == 0:
            continue
        results.append((reverse[int(idx)], float(scores[idx])))
        if len(results) >= top_k:
            break
    return results
