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


def preprocess_comp_dgl(src="data/comp.dgl", data_name="comp", save_dir="data"):
    """FDCompCN (SplitGNN) is already a saved .dgl with its own features/split.
    Repackage it into the LEX-GNN contract (x/y/y_mask + .pt split files),
    reusing SplitGNN's existing split so results stay comparable to their paper."""
    os.makedirs(save_dir, exist_ok=True)
    g, _ = dgl.load_graphs(src)
    g = g[0]

    # Keep only the 3 business relations; drop 'homo' (union) so LEX-GNN sees a
    # clean 3-relation heterograph like Yelp/Amazon and doesn't double-count edges.
    keep = {'invest_bc2bc', 'provide_bc2bc', 'sale_bc2bc'}
    canon = [c for c in g.canonical_etypes if c[1] in keep]
    g = dgl.edge_type_subgraph(g, canon)

    feat = g.ndata['feature'].float()
    label = g.ndata['label'].long()

    idx_train = g.ndata['train_mask'].bool().nonzero(as_tuple=True)[0]
    idx_valid = g.ndata['valid_mask'].bool().nonzero(as_tuple=True)[0]
    idx_test = g.ndata['test_mask'].bool().nonzero(as_tuple=True)[0]

    # LEX-GNN contract: hide every non-train label (2), keep train labels visible.
    y_mask = torch.full_like(label, 2)
    y_mask[idx_train] = label[idx_train]
    y = label.clone()
    y[y < 0] = 0  # defensive: unlabeled(-1) never used as a target, keep index valid

    g.ndata['x'] = feat
    g.ndata['y'] = y
    g.ndata['y_mask'] = y_mask

    dgl.save_graphs(f"{save_dir}/{data_name}_graph.dgl", g)
    torch.save(idx_train, f"{save_dir}/{data_name}_train.pt")
    torch.save(idx_valid, f"{save_dir}/{data_name}_valid.pt")
    torch.save(idx_test, f"{save_dir}/{data_name}_test.pt")
    print(f"comp preprocessing done: {feat.shape[0]} nodes, {feat.shape[1]} feats, "
          f"{len(canon)} relations, train/valid/test = "
          f"{len(idx_train)}/{len(idx_valid)}/{len(idx_test)}")


def _selfcheck_comp(data_name="comp", save_dir="data"):
    """One runnable check: fails loudly if any assumption about comp.dgl breaks."""
    g, _ = dgl.load_graphs(f"{save_dir}/{data_name}_graph.dgl")
    g = g[0]
    x, y, y_mask = g.ndata['x'], g.ndata['y'], g.ndata['y_mask']
    assert x.dim() == 2 and x.dtype == torch.float32, f"x bad: {x.dim()}D {x.dtype}"
    assert set(y.unique().tolist()) <= {0, 1}, f"y not binary: {y.unique().tolist()}"
    assert set(y_mask.unique().tolist()) <= {0, 1, 2}, f"y_mask bad: {y_mask.unique().tolist()}"
    assert len(g.etypes) == 3, f"expected 3 relations, got {g.etypes}"
    tr = set(torch.load(f"{save_dir}/{data_name}_train.pt").tolist())
    va = set(torch.load(f"{save_dir}/{data_name}_valid.pt").tolist())
    te = set(torch.load(f"{save_dir}/{data_name}_test.pt").tolist())
    assert tr and va and te, "empty split"
    assert not (tr & va) and not (tr & te) and not (va & te), "split overlap"
    print(f"selfcheck OK: {x.shape[0]} nodes, {x.shape[1]} feats, etypes={g.etypes}")


def main():
    preprocess_and_save('yelp', seed=30, train_ratio=0.4, test_ratio=0.67)
    preprocess_and_save('amazon', seed=30, train_ratio=0.4, test_ratio=0.67)
    preprocess_comp_dgl()

if __name__ == "__main__":
    main()
    _selfcheck_comp()