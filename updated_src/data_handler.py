import sys
import dgl
import torch
from dgl.dataloading import NeighborSampler, DataLoader

def load_processed_data(graph_path, batch_size, n_layer, num_workers=0, use_cpu_affinity=False):
    graph, _ = dgl.load_graphs(f"../data/{graph_path}_graph.dgl")
    graph = graph[0]

    # Load the custom split indices saved by preprocess_data.py
    idx_train = torch.load(f"../data/{graph_path}_train.pt")
    idx_valid = torch.load(f"../data/{graph_path}_valid.pt")
    idx_test  = torch.load(f"../data/{graph_path}_test.pt")

    n_sample = {e: 50 for e in graph.etypes}
    sampler = NeighborSampler([n_sample] * n_layer)

    train_loader = DataLoader(graph, idx_train, sampler,
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    valid_loader = DataLoader(graph, idx_valid, sampler,
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(graph, idx_test,  sampler,
                              batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    # ponytail: cpu_affinity is POSIX-only; skip on Windows
    if use_cpu_affinity and sys.platform != 'win32':
        train_loader.enable_cpu_affinity()
        valid_loader.enable_cpu_affinity()
        test_loader.enable_cpu_affinity()

    return graph.ndata["x"].shape[1], train_loader, valid_loader, test_loader
