import torch
import random
import numpy as np
import dgl
from dgl.data.fraud import FraudAmazonDataset, FraudYelpDataset
from dgl import RowFeatNormalizer
from sklearn.model_selection import train_test_split
import os


def preprocess_and_save(
    data_name,
    seed=2,
    train_ratio=0.4,
    test_ratio=0.67,
    save_dir="data"
):
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    if data_name == 'yelp':
        graph = FraudYelpDataset().graph
        idx_unlabeled = 0
    else:
        graph = FraudAmazonDataset().graph
        idx_unlabeled = 3305
        transform = RowFeatNormalizer(subtract_min=True, node_feat_names=['feature'])
        graph = transform(graph)

    features = graph.ndata["feature"]
    labels = graph.ndata["label"]

    index = list(range(len(labels)))

    idx_train, idx_rest, _, y_rest = train_test_split(
        index[idx_unlabeled:], labels[idx_unlabeled:],
        stratify=labels[idx_unlabeled:],
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )

    idx_valid, idx_test, _, _ = train_test_split(
        idx_rest, y_rest,
        stratify=y_rest,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True
    )

    # Masking
    y_mask = labels.clone()
    y_mask[index[:idx_unlabeled] + idx_valid + idx_test] = 2

    graph.ndata["x"] = features.float()
    graph.ndata["y"] = labels
    graph.ndata["y_mask"] = y_mask

    # Edge probability
    for etype in graph.canonical_etypes:
        src, _ = graph.edges(etype=etype)
        prob = torch.where(y_mask[src] == 2, 0.5, 0.9)
        graph.edges[etype].data['prob'] = prob

    # Save
    dgl.save_graphs(f"{save_dir}/{data_name}_graph.dgl", graph)
    torch.save(idx_train, f"{save_dir}/{data_name}_train.pt")
    torch.save(idx_valid, f"{save_dir}/{data_name}_valid.pt")
    torch.save(idx_test, f"{save_dir}/{data_name}_test.pt")

    print("Preprocessing selesai & disimpan.")

def main():
    preprocess_and_save('yelp', seed=30, train_ratio=0.4, test_ratio=0.67)
    preprocess_and_save('amazon', seed=30, train_ratio=0.4, test_ratio=0.67)

if __name__ == "__main__":
    main()