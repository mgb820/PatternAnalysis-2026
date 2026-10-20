import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")                      # draw to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score

from dataset import get_data
from modules import GCN, MLP
from train import to_device

# Class names identified by inspecting the page_name field of facebook.npz.
CLASS_NAMES = ["Companies", "Governmental organisations", "Politicians", "Television shows"]


def load_run(run_dir, device):
    """Load the data, split and trained models exactly as used in training."""
    with open(os.path.join(run_dir, "results.json")) as f:
        saved = json.load(f)
    cfg = saved["args"]
    d = to_device(get_data(root=cfg["data_root"], seed=cfg["split_seed"]), device)
    models = {}
    for name, Model in (("MLP", MLP), ("GCN", GCN)):
        model = Model(d["x"].size(1), cfg["hidden"], d["num_classes"],
                      num_layers=cfg["layers"], dropout=cfg["dropout"]).to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, f"{name}.pt"),
                                         map_location=device))
        model.eval()                       # dropout off for inference
        models[name] = model
    return d, models, saved


def predict(model, d):
    """Softmax probabilities and hidden-layer embeddings for every node."""
    with torch.no_grad():
        logits, emb = model(d["x"], d["adj"], return_embedding=True)
    return F.softmax(logits, dim=1), emb


def plot_confusion(y_true, preds, path):
    """Row-normalised confusion matrices (each row sums to 1), with raw counts."""
    k = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, len(preds), figsize=(6.5 * len(preds), 5.5))
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=range(k))
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center",
                        va="center", fontsize=9,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(k))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticks(range(k))
        ax.set_yticklabels(CLASS_NAMES)
        ax.set(xlabel="Predicted category", ylabel="True category",
               title=f"{name} (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tsne(embeddings, y_true, path, max_points=3000, seed=0):
    """2D t-SNE of the last hidden layer for a random sample of test nodes."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), size=min(max_points, len(y_true)), replace=False)
    fig, axes = plt.subplots(1, len(embeddings), figsize=(6.5 * len(embeddings), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (name, emb) in zip(axes, embeddings.items()):
        xy = TSNE(n_components=2, init="pca", perplexity=30,
                  random_state=seed).fit_transform(emb[idx])
        for c, cname in enumerate(CLASS_NAMES):
            mask = y_true[idx] == c
            ax.scatter(xy[mask, 0], xy[mask, 1], s=6, label=cname)
        ax.set_title(f"{name}: hidden-layer embeddings (t-SNE)")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(markerscale=3, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualise trained models.")
    parser.add_argument("--run-dir", default="outputs/seed0",
                        help="folder written by train.py (holds MLP.pt, GCN.pt, results.json)")
    parser.add_argument("--fig-dir", default="figures", help="where to save the figures")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, models, saved = load_run(args.run_dir, device)

    test_idx = d["test_idx"]
    y_test = d["y"][test_idx].cpu().numpy()
    preds, embeddings = {}, {}
    for name, model in models.items():
        probs, emb = predict(model, d)
        y_pred = probs.argmax(1)[test_idx].cpu().numpy()
        preds[name] = y_pred
        embeddings[name] = emb[test_idx].cpu().numpy()
        acc = float((y_pred == y_test).mean())
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        stored = saved["results"][name]
        print(f"{name}: test accuracy {acc:.3f}, macro-F1 {f1:.3f} "
              f"(training run: {stored['accuracy']:.3f}, {stored['macro_f1']:.3f})")

    plot_confusion(y_test, preds, os.path.join(args.fig_dir, "confusion_matrices.png"))
    plot_tsne(embeddings, y_test, os.path.join(args.fig_dir, "tsne_embeddings.png"))
    print(f"Saved figures in '{args.fig_dir}/'")


if __name__ == "__main__":
    main()