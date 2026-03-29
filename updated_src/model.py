import torch
import torch.nn as nn
from dgl.nn import GATv2Conv as ATTNconv


# Node Label Predictor
class NodeLabelPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(NodeLabelPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x): 
        x = self.mlp(x)
        return x


# Node Label Embedding
class NodeLabelEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(NodeLabelEmbedding, self).__init__()
        self.embedding = nn.Embedding(3, embedding_dim, padding_idx =2)  # (0, 1)
    
    def forward(self, label_probs):
        p = label_probs
        emb_0 = self.embedding(torch.zeros_like(p, dtype=torch.long)) 
        emb_1 = self.embedding(torch.ones_like(p, dtype=torch.long)) 
        return (1 - p).unsqueeze(-1) * emb_0 + p.unsqueeze(-1) * emb_1  


# LEX GAT
class LEXGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(LEXGAT, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.node_emb = NodeLabelEmbedding(self.in_dim)
        self.gat = ATTNconv(hidden_dim, hidden_dim, num_heads, allow_zero_in_degree=True)
        self.Ws0 = nn.Linear(self.in_dim, hidden_dim)
        self.Ws1 = nn.Linear(self.in_dim, hidden_dim)
        self.Wd0 = nn.Linear(self.hidden_dim, hidden_dim)
        self.Wd1 = nn.Linear(self.hidden_dim, hidden_dim)
        self.act = nn.ELU()
    
    def forward(self, g, x, y):
        with g.local_scope():
            z_node = self.node_emb(y)
            z = x + z_node
                
            n_dst = g.num_dst_nodes()
            z0 = self.Ws0(z)
            z1 = self.Ws1(z)
            z  = (1 - y).unsqueeze(1) * z0 + y.unsqueeze(1) * z1
    
            h = self.gat(g, z)
            h = torch.mean(h, dim=1)
            h = self.act(h)
                
            p = y[:n_dst]
            h0 = self.Wd0(h)
            h1 = self.Wd1(h)
            h = (1 - p).unsqueeze(1) * h0 + p.unsqueeze(1) * h1
                
            return h


# LEX-GNN layer
class LEXConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(LEXConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(dropout)
        
        self.w_self = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.w_final = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        self.prob_node = NodeLabelPredictor(self.in_dim, self.hidden_dim)
        self.gat = LEXGAT(self.in_dim, self.hidden_dim, num_heads)
    
    def forward(self, block, x, y): 
        
        h = self.feat_drop(x)
        q = self.prob_node(h)
        p = torch.softmax(q, dim=1)[:,1]
        p = torch.where(y==2, p, y.float())
        
        h_self = h[: block.number_of_dst_nodes()]
        h_self = self.w_self(h_self)
        h_conv = [h_self]
        

        for rel in block.etypes:
            h_rel = self.gat(block[rel], h, p)
            h_conv.append(h_rel)
            
        h = torch.stack(h_conv)
        h = torch.sum(h, dim=0)
        h = self.w_final(h)
    
        return h, q


# LEX-GNN model
class LEXGNN(nn.Module):
    def __init__(self, in_dim, n_class, hidden_dim, n_layer, num_heads, dropout):
        super(LEXGNN, self).__init__()
        
        self.in_dim = in_dim
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_layer = n_layer
        self.dropout = dropout
        
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.act = nn.ELU()

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            in_dim = self.in_dim if i==0 else self.hidden_dim
            out_dim = self.hidden_dim
            self.layers.append(LEXConv(in_dim, hidden_dim, out_dim, num_heads, dropout))
            
        self.readout = NodeLabelPredictor(self.hidden_dim, self.n_class)

    
    def forward(self, blocks): 
        n_dst = blocks[-1].number_of_dst_nodes()
        x = blocks[0].srcdata['x'].clone()
        y = blocks[0].srcdata['y_mask'].clone()
        y[:n_dst] = 2 # to avoid label leakage
        
        q_list = []
        for i in range(self.n_layer):
            h, q = self.layers[i](blocks[i], x, y)
            q_list.append(q)
            if i < self.n_layer-1:
                x = self.act(h)
                y = y[:len(x)] 
                
        q = self.readout(h) 
        return q, q_list

