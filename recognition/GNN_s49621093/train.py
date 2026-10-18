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


def train_model(model, d, epochs, lr, weight_decay):
    pass