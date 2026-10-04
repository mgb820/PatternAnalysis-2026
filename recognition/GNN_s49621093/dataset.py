"""Data loading, graph preprocessing and leakage-free splitting.

Setting: TRANSDUCTIVE node classification. The whole graph (all node features
and edges) is visible during training, but labels of validation/test nodes are
never used in the loss. Evaluation uses only the held-out node indices.
"""
import torch
from torch_geometric.datasets import FacebookPagePage


def load_graph(root="data"):
    """Load the Facebook Page-Page graph. Returns (data, num_classes)."""
    dataset = FacebookPagePage(root=f"{root}/FacebookPagePage")
    return dataset[0], dataset.num_classes


def normalized_adjacency(edge_index, num_nodes):
    """Return D^-1/2 (A + I) D^-1/2 as a sparse tensor (Kipf & Welling, 2017)."""
    loops = torch.arange(num_nodes)
    row = torch.cat([edge_index[0], loops])
    col = torch.cat([edge_index[1], loops])
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), torch.ones(row.size(0)), (num_nodes, num_nodes)
    ).coalesce()
    idx = adj.indices()
    vals = torch.ones(idx.size(1))              # binary weights even if edges repeat
    deg = torch.zeros(num_nodes).scatter_add_(0, idx[0], vals)
    d_inv_sqrt = deg.pow(-0.5)
    vals = d_inv_sqrt[idx[0]] * vals * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()


def stratified_split(y, train_frac=0.6, val_frac=0.2, seed=0):
    """Random node split that keeps class proportions the same in each part.

    Returns three disjoint index tensors (train, val, test).
    """
    g = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        n_train = int(train_frac * idx.numel())
        n_val = int(val_frac * idx.numel())
        train.append(idx[:n_train])
        val.append(idx[n_train:n_train + n_val])
        test.append(idx[n_train + n_val:])
    return torch.cat(train), torch.cat(val), torch.cat(test)


def get_data(root="data", train_frac=0.6, val_frac=0.2, seed=0):
    """Everything the training script needs, in one dict."""
    data, num_classes = load_graph(root)
    adj = normalized_adjacency(data.edge_index, data.num_nodes)
    train_idx, val_idx, test_idx = stratified_split(data.y, train_frac, val_frac, seed)
    return {
        "x": data.x, "y": data.y, "adj": adj, "edge_index": data.edge_index,
        "num_classes": num_classes,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


if __name__ == "__main__":
    d = get_data()
    print("nodes:", d["x"].size(0), "| features:", d["x"].size(1),
          "| classes:", d["num_classes"])
    print("train/val/test:", len(d["train_idx"]), len(d["val_idx"]), len(d["test_idx"]))