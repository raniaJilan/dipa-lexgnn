import torch
import random
import numpy as np
import dgl
from dgl.data.fraud import FraudAmazonDataset, FraudYelpDataset
from dgl.dataloading import NeighborSampler, DataLoader
from dgl import RowFeatNormalizer
from sklearn.model_selection import train_test_split


def load_data(data_name, seed, train_ratio, test_ratio, n_layer, batch_size):

    # Load dataset
    if data_name == 'yelp':
        graph = FraudYelpDataset().graph
        node = 'review'
        idx_unlabeled = False
    else:
        graph = FraudAmazonDataset().graph
        node = 'user'
        idx_unlabeled = 3305
        transform = RowFeatNormalizer(subtract_min=True, node_feat_names=['feature'])
        graph = transform(graph)

    features = graph.ndata["feature"]
    labels = graph.ndata["label"]
    
    # Data split
    index = list(range(len(labels)))
    idx_train, idx_rest, y_train, y_rest = train_test_split(index[idx_unlabeled:], labels[idx_unlabeled:], 
                                                            stratify=labels[idx_unlabeled:], 
                                                            train_size=train_ratio, random_state=seed, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, 
                                                            test_size=test_ratio, random_state=seed, shuffle=True)

    # Masking unlabeled nodes
    graph.ndata["y"] = labels
    y_mask = labels.clone() 
    y_mask[index[:idx_unlabeled]+idx_test+idx_valid] = 2 
    graph.ndata["y_mask"] = y_mask
    graph.ndata["x"] = torch.FloatTensor(features).contiguous()

    print(data_name.upper(), len(idx_train), len(idx_valid), len(idx_test))
    
    # Batch loader
    n_sample = {}
    for e in graph.etypes:
        n_sample[e] = 50
    
    n_samples = [n_sample]*n_layer
    
    edge_probs = {}
    for etype in graph.canonical_etypes:
        src, dst = graph.edges(etype=etype)
        prob = torch.where(y_mask[src] == 2, 0.5, 0.9) 
        edge_probs[etype] = prob
        graph.edges[etype].data['prob'] = prob
            
    sampler = NeighborSampler(n_samples, prob='prob')
    train_loader = DataLoader(graph, idx_train, sampler, batch_size=batch_size, shuffle=True, drop_last=False, use_uva=True)
    valid_loader = DataLoader(graph, idx_valid, sampler, batch_size=batch_size, shuffle=False, drop_last=False, use_uva=True)
    test_loader = DataLoader(graph, idx_test, sampler, batch_size=batch_size, shuffle=False, drop_last=False, use_uva=True)

    return features.shape[1], train_loader, valid_loader, test_loader



