from model_MCGAT import MCGAT, LinkPredictor
import torch
import torch.nn as nn
import dgl
import pandas as pd
import numpy as np
import itertools
from sklearn.metrics import roc_auc_score
import re
import os

def extract_numeric(s):
    matches = re.findall(r'\d+', s)
    return int(matches[0]) if matches else None

def read_file(filename, directory = 'data'):
    f = os.path.join(directory, filename)
    print(filename)
    if os.path.isfile(f):
        df = pd.read_csv(f)
        # Extract only digits from name ('GP000002' to 2)
        df['source']= df['source'].apply(lambda x: extract_numeric(x))
        df['target']= df['target'].apply(lambda x: extract_numeric(x))
    return df

def to_idx_tensor(df, source_idx, target_idx, rev=False):
    a= df['source'].apply(lambda x: source_idx[x])
    b= df['target'].apply(lambda x: target_idx[x])
    source_tensor = torch.tensor(a.values)
    target_tensor = torch.tensor(b.values)

    if rev:
        return target_tensor, source_tensor
    return source_tensor, target_tensor

def score(pos_score, neg_score):
    pos_score = pos_score.cpu()
    neg_score = neg_score.cpu()
    acc = (torch.sum(neg_score < 0.5).item() + torch.sum(pos_score > 0.5).item())/(len(pos_score)+len(neg_score))
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    return roc_auc_score(labels, scores), acc


def criterion(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    loss_func = nn.BCELoss()
    return loss_func(scores, labels)

def get_weights(g): 
    src_weights = g.out_degrees(etype=etype).float()
    src_weights = src_weights.pow(0.75)
    src_weights /= src_weights.sum()
    dst_weights = g.in_degrees(etype=etype).float()
    dst_weights = dst_weights.pow(0.75)
    dst_weights /= dst_weights.sum()
    
    return src_weights, dst_weights

def weighted_neg_sampler(num_negs, k, src_weights, dst_weights):
    neg_edges = set()
    
    num_negs = k*num_negs
    
    while len(neg_edges) < num_negs:
        potential_neg_src = torch.multinomial(src_weights, num_negs, replacement=True)
        potential_neg_dst = torch.multinomial(dst_weights, num_negs, replacement=True)
        potential_neg_edges = set(zip(potential_neg_src.cpu().numpy(), potential_neg_dst.cpu().numpy()))

        # Remove positive edges
        potential_neg_edges.difference_update(all_edges)
        neg_edges.update(potential_neg_edges)

    neg_edges = list(neg_edges)[:num_negs]
    neg_src, neg_dst = zip(*neg_edges)
    neg_src = torch.tensor(neg_src)
    neg_dst = torch.tensor(neg_dst)
    return neg_src, neg_dst






#####Graph Construction#####

print("-------------Graph Construction-------------")
# read files
cp_cp_df = read_file("cp_cp_id.csv")
cp_ph_df = read_file("cp_ph_id.csv")
ph_ph_df = read_file("ph_ph_id.csv")
he_cp_df = read_file("coconut_he_cp.csv")
he_ph_df = read_file("coconut_he_ph.csv")

# Sort the IDs to assign indices of cp, gp, bp, ph
cp_sorted = sorted(list(set(cp_cp_df['source'].values.tolist() + cp_cp_df['target'].values.tolist() + cp_ph_df['source'].values.tolist() + he_cp_df['target'].values.tolist())))
ph_sorted = sorted(list(set(cp_ph_df['target'].values.tolist() + ph_ph_df['source'].values.tolist() + ph_ph_df['target'].values.tolist() + he_ph_df['target'].values.tolist())))
he_sorted = sorted(list(set(he_cp_df['source'].values.tolist() + he_ph_df['source'].values.tolist())))

# make dictionary where the key is the orginal ID and the value is the index of node.
cp_idx = {value: index for index, value in enumerate(cp_sorted)}
ph_idx = {value: index for index, value in enumerate(ph_sorted)}
he_idx = {value: index for index, value in enumerate(he_sorted)}
cp_idx_rev = {index: value for index,value in enumerate(cp_sorted)}
ph_idx_rev = {index: value for index, value in enumerate(ph_sorted)}
he_idx_rev = {index: value for index, value in enumerate(he_sorted)}


g = dgl.heterograph(
        {
            ("compound", "cc", "compound"): to_idx_tensor(cp_cp_df,cp_idx,cp_idx),
            ("compound", "cp", "phenotype"): to_idx_tensor(cp_ph_df,cp_idx,ph_idx),
            ("phenotype", "pp", "phenotype"): to_idx_tensor(ph_ph_df,ph_idx,ph_idx),
            ("herb", "hc", "compound"): to_idx_tensor(he_cp_df,he_idx,cp_idx),
            ("herb", "hp", "phenotype"): to_idx_tensor(he_ph_df,he_idx,ph_idx),
            ("compound", "ch", "herb"): to_idx_tensor(he_cp_df,he_idx,cp_idx, rev=True),
            ("phenotype", "ph", "herb"): to_idx_tensor(he_ph_df,he_idx,ph_idx, True),
            ("phenotype", "pc", "compound"): to_idx_tensor(cp_ph_df,cp_idx,ph_idx, True),
        }
    )
g.add_edges(*to_idx_tensor(cp_cp_df,cp_idx,cp_idx,True),etype="cc")
g.add_edges(*to_idx_tensor(ph_ph_df,ph_idx,ph_idx,True),etype="pp")

g = g.cpu()
# Specify the dimensionality of node features
feature_dim = 1024

# Assign random node features to each node type
for ntype in g.ntypes:
    num_nodes = g.number_of_nodes(ntype)
    # Generate random features follwing standard normal distribution
    random_features = torch.randn(num_nodes, feature_dim)
    # Assign the random features to the nodes
    g.nodes[ntype].data['features'] = random_features



device = torch.device('cuda')
#device = torch.device('cpu')
device

hidden_size = 128
num_heads = [8]
feature_dim = 1024

pred = LinkPredictor(in_channels = num_heads[0] * hidden_size * 2)
pred = pred.to(device)


##### Generate metapaths list #####
print("-------------Generate metapaths list-------------")

chars = ['c', 'p', 'h']
test_dict = {
    'herb': [("ch")],
    'compound': [("cc")],
    'phenotype': [("pp")]
    }

# Generate all metapaths from length 2 to 3
all_combinations = []
for length in range(2, 4):
    all_combinations.extend(itertools.product(chars, repeat=length))

# Filter out unvalid metapaths
valid_combinations = []
not_valid = ['hh']
for comb in all_combinations:
    joined = ''.join(comb)
    app = True
    for n in not_valid:
        if n in joined:
            app = False
            break
    if app:
        valid_combinations.append(joined)

# Results
for comb in valid_combinations:    
    if comb[-1] == 'c':
        if len(comb) == 3:
            comb = (comb[0] + comb[1], comb[1] + comb[2])
        elif len(comb) == 4:
            comb = (comb[0] + comb[1], comb[1] + comb[2], comb[2] + comb[3])
        else:
            comb = (comb[0] + comb[1])
            
        if comb not in test_dict['compound']:
            test_dict['compound'].append(comb)

    elif comb[-1] == 'p':
        if len(comb) == 3:
            comb = (comb[0] + comb[1], comb[1] + comb[2])
        elif len(comb) == 4:
            comb = (comb[0] + comb[1], comb[1] + comb[2], comb[2] + comb[3])
        else:
            comb = (comb[0] + comb[1])
        
        if comb not in test_dict['phenotype']:
            test_dict['phenotype'].append(comb)
            
    elif comb[-1] == 'h':
        if len(comb) == 3:
            comb = (comb[0] + comb[1], comb[1] + comb[2])
        elif len(comb) == 4:
            comb = (comb[0] + comb[1], comb[1] + comb[2], comb[2] + comb[3])
        else:
            comb = (comb[0] + comb[1])
        if comb not in test_dict['herb']:
            test_dict['herb'].append(comb)
            
test_list = ['base']
test_list += (test_dict['herb'])
test_list += (test_dict['compound'])
test_list += (test_dict['phenotype'])

# remove base metapaths
test_list.remove('ch')
test_list.remove('cc')
test_list.remove('pp')


print(test_list)

print()

print("-------------Test all metapaths-------------")
# Test all metapaths

results = []

for i in range(5):
    etype = 'hp'
    rev_etype = 'ph'
    stype = 'herb'
    dtype = 'phenotype'
    
#     Split into train/val/test
    eids = np.arange(g[etype].number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids)*0.2)
    val_size = int(len(eids)*0.2)

    train_size = g[etype].number_of_edges() - test_size - val_size

    train_g = dgl.remove_edges(g, eids[:test_size + val_size],etype)
    test_g = dgl.remove_edges(g, eids[test_size:],etype)
    val_g = dgl.remove_edges(g, np.concatenate((eids[:test_size],eids[test_size + val_size:])),etype)
    train_g = dgl.remove_edges(train_g, eids[:test_size + val_size],rev_etype)
    test_g = dgl.remove_edges(test_g, eids[test_size:],rev_etype)
    val_g = dgl.remove_edges(val_g, np.concatenate((eids[:test_size],eids[test_size + val_size:])),rev_etype)

    train_features = train_g.ndata['features']
    train_features = {key: tensor.to(device) for key, tensor in train_features.items()}
    val_features = val_g.ndata['features']
    val_features = {key: tensor.to(device) for key, tensor in val_features.items()}
    test_features = test_g.ndata['features']
    test_features = {key: tensor.to(device) for key, tensor in test_features.items()}

    train_g = train_g.to(device)
    val_g = val_g.to(device)
    test_g = test_g.to(device)

    pos_edges = g.edges(etype=etype)
    all_edges = set(zip(pos_edges[0].cpu().numpy(), pos_edges[1].cpu().numpy()))

    train_pos_edges = train_g.edges(etype=etype)
    train_num_pos = len(train_pos_edges[0])
    val_pos_edges = val_g.edges(etype=etype)
    val_num_pos = len(val_pos_edges[0])
    test_pos_edges = test_g.edges(etype=etype)
    test_num_pos = len(test_pos_edges[0])
    
    train_src_weights, train_dst_weights = get_weights(train_g)
    val_src_weights, val_dst_weights = get_weights(val_g)
    test_src_weights, test_dst_weights = get_weights(test_g)
    train_neg_edges = weighted_neg_sampler(train_num_pos,1,train_src_weights,train_dst_weights)
    val_neg_edges = weighted_neg_sampler(val_num_pos,1,val_src_weights,val_dst_weights)
    test_neg_edges = weighted_neg_sampler(test_num_pos,1,test_src_weights,test_dst_weights)
    

    for path in test_list:
        metapath_dict = {
        'herb': [("ch")],
        'compound': [("cc")], 
        'phenotype': [("pp")],
        }
        if path[-1][-1] == 'c':
            ntype = 'compound'
            metapath_dict[ntype] = metapath_dict[ntype] + [path]
        elif path[-1][-1] == 'p':
            ntype = 'phenotype'
            metapath_dict[ntype] = metapath_dict[ntype] + [path]
        elif path != 'base':
            ntype = 'herb'
            metapath_dict[ntype] = metapath_dict[ntype] + [path]
        
        model = MCGAT(
            meta_paths= metapath_dict,
            ntypes = ['compound','herb','phenotype'],
            in_size=feature_dim,
            hidden_size=hidden_size,
            num_heads=[8],
            dropout=0.6,
            update_cnt=1,
        ).to(device)
        g = g.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        # Train
        best_val_auc = 0.5
        best_epoch = 0

        for epoch in range(100):
            model.train()
            h = model(train_g, train_features)
            pos_score, neg_score = pred(h[stype][train_pos_edges[0]], h[dtype][train_pos_edges[1]]), pred(h[stype][train_neg_edges[0]], h[dtype][train_neg_edges[1]])

            pos_score = pos_score.to(device)
            neg_score = neg_score.to(device)
            loss = criterion(pos_score, neg_score)
            train_auc, train_acc = score(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_h = model(val_g, val_features)
                pos_score, neg_score = pred(val_h[stype][val_pos_edges[0]], val_h[dtype][val_pos_edges[1]]), pred(val_h[stype][val_neg_edges[0]], val_h[dtype][val_neg_edges[1]])
            val_loss = criterion(pos_score, neg_score)
            val_auc, val_acc = score(pos_score, neg_score)

            if val_auc > best_val_auc:
                best_epoch = epoch
                best_val_auc = val_auc
                torch.save(model.state_dict(), "best_model_combi_he.pth")


        print("Path: "+str(path))
        print(
            "End Epoch {:d} | Best Epoch {:d} | Best valauc {:.4f} ".format(
                epoch + 1,
                best_epoch + 1,
                best_val_auc,
            )
        )

        model.load_state_dict(torch.load("best_model_combi_he.pth"))
        model.eval()

        with torch.no_grad():
            test_h = model(test_g, test_features)
            pos_score, neg_score = pred(test_h[stype][test_pos_edges[0]], test_h[dtype][test_pos_edges[1]]), pred(test_h[stype][test_neg_edges[0]], test_h[dtype][test_neg_edges[1]])


        test_loss = criterion(pos_score, neg_score)
        test_auc, test_acc = score(pos_score, neg_score)

        print(
            "Test Loss {:.4f} | Test AUC {:} | Test Acc {:.4f} ".format(
                test_loss.item(),
                test_auc,
                test_acc,
            )
        )
        results.append([path,i+1,test_auc])
        
    del train_g, val_g, test_g, train_features, val_features, test_features
    torch.cuda.empty_cache()
    print(results)
    
print()


rows=[]
for j in range(len(test_list)):
    want = []
    for i in range(5):
        want.append(results[i*len(test_list)+j][2])
    want.append(float(np.average(want)))
    want.append(results[i*len(test_list)+j][0])
    rows.append(want)

pd.DataFrame(rows).to_csv("metapath_selection.csv")