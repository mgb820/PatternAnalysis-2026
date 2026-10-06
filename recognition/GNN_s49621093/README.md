Dataset: Facebook Large Page-Page Network, loaded with PyTorch Geometric's FacebookPagePage class.
- 22480 pages
- 128 dimensional node features
- 4 categories (politicians, povernmental organisations, television shows, companies)

Transductive node classification (full graph is visible during training, and only the labels of training nodes are used in the loss)

Edges are treated as undirected, and the adjacency matrix is symmetrically normalised.
Using this equation Â = D^(-1/2)(A + I)D^(-1/2) (Kipf and Welling 2017).

Splits: Nodes are split 60% train, 20% validation, and 20% test. Stratified by class so each split has same class proportions are the whole thing.


Feasibility Check-Off:
(Written in word document on personal computer and then copy pasted in):

1. User Need, Scope & Acceptance Criteria
User: A web platform or data research team that needs to categorise Facebook pages automatically, with low-confidence predictions routed to a human reviewer.
Scope: Semi-supervised, 4-class node classification on the Facebook Page-Page graph, comparing a multi-layer GNN with an MLP baseline on identical splits.
Acceptance criteria (1-3 on the held-out test split): (1) GNN macro-F1 is at least 0.05 above the MLP's. (2) Mean softmax confidence on correct predictions exceeds that on incorrect ones by at least 0.25, and confidence detects errors with AUROC of at least 0.85. (3) A threshold chosen on the validation set gives at least 95% test accuracy above it while covering at least 60% of nodes; the rest go to human review. (4) Training takes under 5 minutes and peak GPU memory stays under 2 GB (provisional) on one GPU.
2. Model Choice & Course Concepts
Data: Facebook Large Page-Page Network: 22,470 pages, mutual-like edges, 128-dimensional features, 4 categories. The task is transductive: the whole graph is visible in training, but only training labels enter the loss. Nodes are split 60/20/20, stratified by class with a fixed seed. Validation data is used for model selection and the threshold; test data only for final reporting.
Model: A 2-layer GCN1 with hidden size 64 and dropout 0.5, in PyTorch (Normal difficulty). The baseline is an MLP of the same size using node features only.
Concepts: Each GCN layer averages a node's features with its neighbours' (message passing), which should help because linked pages tend to share a category. The normalised adjacency with self-loops keeps a node's own features and stops high-degree pages dominating. Shallow depth avoids over-smoothing, and dropout limits overfitting. Softmax probability is the confidence signal for the reject rule.
3. Preliminary Feasibility Evidence
Data audit: 22,470 nodes, 128 features, 4 classes (28.9%, 30.6%, 25.7%, 14.8%); no isolated nodes; mean degree 15.2; edge homophily 0.885. A 2-layer MLP trained for 100 epochs on a 60/20/20 random split reached test accuracy 0.793 and macro-F1 0.774. Mean confidence was 0.816 on correct and 0.556 on incorrect predictions (error-detection AUROC 0.823). A threshold of 0.8, chosen on validation data, gave 95.4% test accuracy on the 51% of nodes retained. GPU use (MLP, one A100): 3.4 s for 100 epochs (0.034 s/epoch), peak memory 49.8 MB. The pipeline loads, trains and evaluates end to end, with headroom for a GNN to improve on the baseline.
4. Risks, Budget & Fallback
Risks: label leakage through the graph (loss uses training indices only); the GNN failing to beat the MLP or being more overconfident (this is what the criteria test); class imbalance (stratified splits, macro-F1); over-smoothing (2-3 layers); and limited GPU access (runs kept short).
Budget: The graph and models are small. The MLP trains in 3.4 s with 49.8 MB peak GPU memory, and I expect the GCN to need seconds to a few minutes per run and well under 500 MB. I can measure these values by performing several jobs on one GPU. Peak memory and timing will be measured. Next task is to Implement the GCN, train it with the MLP's loop and split, and compare macro-F1, confidence gap, AUROC and the reject rule.
Fallback: If the GCN does not improve on the MLP, report that and analyse why (for example, accuracy on high- versus low-homophily nodes).
