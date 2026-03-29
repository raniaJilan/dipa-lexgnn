import dgl
import torch
from dgl.dataloading import NeighborSampler, DataLoader


def load_processed_data(graph_path, batch_size, n_layer):
    graph, _ = dgl.load_graphs(f"../data/{graph_path}_graph.dgl")
    graph = graph[0]

    idx_train = torch.nonzero(graph.ndata["train_mask"], as_tuple=True)[0]
    idx_valid = torch.nonzero(graph.ndata["val_mask"],   as_tuple=True)[0]
    idx_test  = torch.nonzero(graph.ndata["test_mask"],  as_tuple=True)[0]

    n_sample = {e: 50 for e in graph.etypes}
    sampler = NeighborSampler([n_sample] * n_layer)

    train_loader = DataLoader(graph, idx_train, sampler,
                              batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(graph, idx_valid, sampler,
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(graph, idx_test, sampler,
                              batch_size=batch_size, shuffle=False)

    return graph.ndata["x"].shape[1], train_loader, valid_loader, test_loader
