import copy
import time

import torch
import torch.nn.functional as F


def to_device(d, device):
    """Move every tensor in the data dict (including the sparse adjacency) to the device."""
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in d.items()}


def evaluate(model, d, idx):
    """Loss, accuracy and softmax probabilities for the nodes in idx."""
    model.eval()
    with torch.no_grad():
        logits = model(d["x"], d["adj"])[idx]
        loss = F.cross_entropy(logits, d["y"][idx]).item()
        probs = F.softmax(logits, dim=1)
        acc = (probs.argmax(1) == d["y"][idx]).float().mean().item()
    return loss, acc, probs


def train_model(model, d, epochs=200, lr=0.01, weight_decay=5e-4):
    """Train one model with Adam and cross-entropy on the training nodes only.

    The weights from the epoch with the best VALIDATION accuracy are restored
    at the end. Returns (history, train_seconds, peak_gpu_mb).
    """
    device = next(model.parameters()).device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state = -1.0, None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(d["x"], d["adj"])
        loss = F.cross_entropy(out[d["train_idx"]], d["y"][d["train_idx"]])
        loss.backward()
        opt.step()

        val_loss, val_acc, _ = evaluate(model, d, d["val_idx"])
        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if device.type == "cuda":
        torch.cuda.synchronize()          # wait for the GPU before stopping the clock
    train_seconds = time.time() - start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None

    model.load_state_dict(best_state)
    return history, train_seconds, peak_mb