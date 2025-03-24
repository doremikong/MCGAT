import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import numpy as np

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin = torch.nn.Linear(in_channels, 1)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()
    
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        

    def __call__(self, score):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False    


class MetapathAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(MetapathAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        sem_att = (beta * z).sum(1)

        return sem_att


class MSGATLayer(nn.Module):
    def __init__(self, meta_paths, ntypes, in_size, out_size, layer_num_heads, dropout, update_cnt, sem_hidden):
        super(MSGATLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleDict()
        self.norm_layers = nn.ModuleDict()
        self.ntypes = ntypes
        self.update_cnt = update_cnt
        for ntype in ntypes:
            self.norm_layers[ntype] = nn.LayerNorm(out_size * layer_num_heads)
            self.gat_layers[ntype] = nn.ModuleList()
            for i in range(len(meta_paths[ntype])):
                self.gat_layers[ntype].append(
                    GATConv(
                        in_size,
                        out_size,
                        layer_num_heads,
                        dropout,
                        dropout,
                        activation=F.elu,
                        allow_zero_in_degree=True,
                    )
                )
        self.metapath_attention = MetapathAttention(
            in_size=out_size * layer_num_heads, hidden_size = sem_hidden
        )
        self.meta_paths = meta_paths

        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.meta_path_ntypes = {}

    def forward(self, g, h):
        metapath_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g: # To calculate only once at first
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for i, ntype in enumerate(self.ntypes):
                for meta_path in self.meta_paths[ntype]:
                    if type(meta_path) == str: # for 1-hop metapath
                        self._cached_coalesced_graph[
                            meta_path
                        ] = g.edge_type_subgraph([meta_path])
                        srctype = g.to_canonical_etype(meta_path)[0]
                        dsttype = g.to_canonical_etype(meta_path)[2]
                    else:
                        self._cached_coalesced_graph[
                            meta_path
                        ] = dgl.metapath_reachable_graph(g, meta_path)
                        srctype = g.to_canonical_etype(meta_path[0])[0]
                        dsttype = g.to_canonical_etype(meta_path[-1])[2]
                    self.meta_path_ntypes[meta_path] = (srctype, dsttype)
        
        new_h = {}
        for i, ntype in enumerate(self.ntypes):
            new_h[ntype] = h[ntype]
        
        for k in range(self.update_cnt):
            for i, ntype in enumerate(self.ntypes):
                for i, meta_path in enumerate(self.meta_paths[ntype]):
                    new_g = self._cached_coalesced_graph[meta_path]
                    srctype, dsttype = self.meta_path_ntypes[meta_path]
                    metapath_embeddings.append(self.gat_layers[ntype][i](new_g, (new_h[srctype], new_h[dsttype])).flatten(1))
                    
                metapath_embeddings = torch.stack(
                    metapath_embeddings, dim=1
                )
                # residual connection (original feature + result of metapath-level attention)
                # layer normalization
                new_h[ntype] = self.norm_layers[ntype](new_h[ntype]) + self.metapath_attention(metapath_embeddings)
                metapath_embeddings = []

        return new_h
    

class MSGAT(nn.Module):
    def __init__(
        self, meta_paths, ntypes, in_size, hidden_size, num_heads, dropout, update_cnt, sem_hidden
    ):
        super(MSGAT, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            MSGATLayer(meta_paths, ntypes, in_size, hidden_size, num_heads[0], dropout, update_cnt, sem_hidden)
        )
        for l in range(1, len(num_heads)): # if try to test multiple num_heads (ex. num_heads = [8,16,4,...])
            self.layers.append(
                MSGATLayer(
                    meta_paths,
                    ntypes,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                    update_cnt,
                    sem_hidden
                )
            )
        self.pred = LinkPredictor(in_channels = num_heads[0] * hidden_size * 2)

    def forward(self, g, h, stype, dtype, pos_edges, neg_edges, tune = False):
        for gnn in self.layers:
            h = gnn(g, h)
        
        if tune == True:
            return h
        
        return self.pred(h[stype][pos_edges[0]], h[dtype][pos_edges[1]]), self.pred(h[stype][neg_edges[0]], h[dtype][neg_edges[1]])