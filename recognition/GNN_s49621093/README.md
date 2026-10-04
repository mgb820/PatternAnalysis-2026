Dataset: Facebook Large Page-Page Network, loaded with PyTorch Geometric's FacebookPagePage class.
- 22480 pages
- 128 dimensional node features
- 4 categories (politicians, povernmental organisations, television shows, companies)

Transductive node classification (full graph is visible during training, and only the labels of training nodes are used in the loss)

Edges are treated as undirected, and the adjacency matrix is symmetrically normalised.
Using this equation Â = D^(-1/2)(A + I)D^(-1/2) (Kipf and Welling 2017).

Splits: Nodes are split 60% train, 20% validation, and 20% test. Stratified by class so each split has same class proportions are the whole thing.
