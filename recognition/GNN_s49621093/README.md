# Node Classification of Facebook Pages with a Graph Convolutional Network

## Overview
Given a graph with Facebook Pages as nodes and edges between pages as likes, the problem is trying to classify these pages into 4 distinct categories. I am using a 2-layer GCN to classify these pages.
The algorithm is a 2-layer GCN. Each layer averages a node's features with its neighbours', which would help with classification because linked pages tend to share a category. The normalised adjacency with self-loops keeps a node's own features and stops high-degree pages dominating. Shallow depth avoids over-smoothing, and dropout limits overfitting. Softmax probability is the confidence signal for the reject rule.

---

## Feasibility Review

### 1. User Need, Scope & Acceptance Criteria

A possible user for this model can be a web platform or data research team that needs to categorise Facebook pages automatically, with low-confidence predictions routed to a human reviewer. The project involves a semi-supervised, 4-class node classification on the Facebook Page-Page graph, comparing a multi-layer GNN with an MLP baseline on identical splits.

**Acceptance criteria:**

| # | Criterion | Measured on |
|---|-----------|-------------|
| 1 | GNN macro-F1 is at least 0.05 above the MLP's | Test split |
| 2 | Mean softmax confidence on correct predictions exceeds that on incorrect ones by at least 0.25, and confidence detects errors with AUROC of at least 0.85 | Test split |
| 3 | A threshold chosen on the validation set gives at least 95% test accuracy above it while covering at least 60% of nodes while the rest go to human review | Test split |
| 4 | Training takes under 5 minutes and peak GPU memory stays under 2 GB (provisional) | One GPU |

### 2. Model Choice & Course Concepts

The data is the Facebook Large Page-Page Network, which has 22,470 pages with mutual-like edges. Each page has 128-dimensional features and are one of four categories. The task is transductive, which means that the whole graph is visible in training, but test labels are not used. Nodes are split 60/20/20 (train, validation, test), stratified by class with a fixed seed. Validation data is used for model selection and the threshold, and the test data is used for final reporting.
The model is a 2-layer GCN1 with hidden size 64 and dropout 0.5 implemented with PyTorch (Normal difficulty). The baseline is an MLP of the same size using node features only.
Each GCN layer averages a node's features with its neighbours', which should help because linked pages tend to share a category. The normalised adjacency with self-loops keeps a node's own features and stops high-degree pages dominating. Shallow depth avoids over-smoothing, and dropout limits overfitting. Softmax probability is the confidence signal for the reject rule.

### 3. Preliminary Feasibility Evidence

**Data audit:** 22,470 nodes, 128 features, 4 classes (28.9%, 30.6%, 25.7%, 14.8%). No isolated nodes, mean degree 15.2, and edge homophily 0.885.

**MLP baseline** (2 layers, 100 epochs, 60/20/20 random split):

| Metric | Value |
|--------|-------|
| Test accuracy | 0.793 |
| Test macro-F1 | 0.774 |
| Mean confidence (correct / wrong) | 0.816 / 0.556 |
| Error-detection AUROC | 0.823 |
| Threshold 0.8 (chosen on validation) | 95.2% test accuracy on the 51% of nodes retained |

**GPU use (MLP, one A100):** 3.4 s for 100 epochs (0.034 s/epoch), peak memory 49.8 MB.

The pipeline loads, trains and evaluates end to end, with headroom for a GNN to improve on the baseline.

### 4. Risks, Budget & Fallback

The risks for the GNN implementation include (1) Label leakage through the graph. (2) The GNN failing to beat the MLP or being more overconfident. (3) Class imbalance (stratified splits, macro-F1). (4) Over-smoothing (which is why we will use only 2-3 layers). (5) Limited GPU access.
As for the budget, the MLP trains in 3.4 s with 49.8 MB peak GPU memory, so the GCN is expected to need no more than a couple of minutes per run and well under 500 MB. Several jobs on one GPU and the job’s peak memory and timing will be measured.
Next task is to Implement the GCN, train it with the MLP's loop and split, and compare macro-F1, confidence gap, AUROC and the reject rule.
If the GCN does not improve on the MLP, that will be reported and analysed (for example, accuracy on high- versus low-homophily nodes).

---

## Setup and Usage

### Dependencies

| Package | Version |
|---------|---------|
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.1 |

Experiments were run on a single NVIDIA A100 (Rangpur) with the `torch` conda environment.

### Data

The Facebook Page-Page graph is read from `facebook.npz`, which was supplied
by the course coordinators because the original download host was unavailable.

### Reproducibility

All splits and model initialisation use a fixed seed (0).

---

## Data and Preprocessing

Dataset details are in the Feasibility Review (part 2). The data is read directly from `facebook.npz`.

**Graph preprocessing:** Edges are treated as undirected, and the adjacency matrix is symmetrically normalised with self-loops (Kipf and Welling [2]):

$$\hat{A} = D^{-1/2}(A + I)D^{-1/2}$$

**Splits:** Nodes are split 60% train, 20% validation and 20% test, stratified by class so that each split has the same class proportions as the whole graph.

**Leakage prevention:** The setting is transductive, so the full graph is visible during training, but only the labels of training nodes enter the loss.

---

## Results

(add results after running the actual thing)

---

## Artificial Intelligence Usage Disclosure

I used Claude Sonnet 5.5 to explain concepts and to help write technical parts of the code. I verified its output through my own research and by running jobs to test the results.

## References

1. B. Rozemberczki, C. Allen and R. Sarkar, "Multi-scale attributed node embedding," *Journal of Complex Networks*, vol. 9, no. 2, 2021.
2. T. N. Kipf and M. Welling, "Semi-supervised classification with graph convolutional networks," in *Proc. ICLR*, 2017.
