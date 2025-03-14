from model_MCGAT import MCGAT, LinkPredictor, EarlyStopping
import torch
import torch.nn as nn
import dgl
import pandas as pd
import re
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, matthews_corrcoef, precision_score, recall_score
import random
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

def get_best_threshold(pos_score,neg_score):
    pos_score = pos_score.cpu()
    neg_score = neg_score.cpu()
    y_prob = torch.cat([pos_score, neg_score]).detach().numpy()
    y_true = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    thresholds = np.linspace(0, 1, 101)

    best_threshold_mcc = 0.0
    best_mcc = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold_mcc = threshold

    return best_threshold_mcc


def test_score(pos_score, neg_score, val_pos_score, val_neg_score):
    pos_score = pos_score.cpu()
    neg_score = neg_score.cpu()
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    precisions, recalls, _ = precision_recall_curve(labels,scores)
    auprc = auc(recalls, precisions)
    threshold = get_best_threshold(val_pos_score,val_neg_score)
    y_pred = (scores >= threshold).astype(int)
    f1 = f1_score(labels, y_pred)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred)
    mcc = matthews_corrcoef(labels, y_pred)
    return roc_auc_score(labels, scores), auprc, f1, precision, recall, mcc

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

def masking(g):
    ne = g[etype].number_of_edges()
    masked = random.sample(range(ne), int(ne*0.1))
    masked_g = dgl.remove_edges(g,masked,etype)
    masked_g = dgl.remove_edges(masked_g,masked,rev_etype)
    return masked_g  


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





device = torch.device('cuda')
#device = torch.device('cpu')
device


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





#####Model#####
metapath_dict = {
    'herb': [("ch")],
    'compound': [("cc")],
    'phenotype': [("pp")],
    }

# metapath_dict = {
#     'herb': [("ch"),("ph"),("pp","ph")],
#     'compound': [("cc")],
#     'phenotype': [("pp"),("cp")],
#     }

hidden_size = 128
num_heads = [8]
feature_dim = 1024
model = MCGAT(
    meta_paths= metapath_dict,
    ntypes = g.ntypes,
    in_size=feature_dim,
    hidden_size=hidden_size,
    num_heads=num_heads,
    dropout=0.6,
    update_cnt=3
).to(device)

g = g.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
es = EarlyStopping(patience=300, mode='max', verbose=False)

pred = LinkPredictor(in_channels = num_heads[0] * hidden_size * 2)
pred = pred.to(device)

etype = 'hp'
rev_etype = 'ph'
stype = 'herb'
dtype = 'phenotype'
# etype = 'cp'
# rev_etype = 'pc'
# stype = 'compound'
# dtype = 'phenotype'





##### Train/Val/Test Split #####
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

val_src_weights, val_dst_weights = get_weights(val_g)
test_src_weights, test_dst_weights = get_weights(test_g)






##### Train #####
print("-------------Train-------------")
best_val_auc = 0.5
best_epoch = 0
losses = []
for epoch in range(5000):
    model.train()

    masked_train_g = masking(train_g)
    train_src_weights, train_dst_weights = get_weights(masked_train_g)
    train_neg_edges = weighted_neg_sampler(train_num_pos,1,train_src_weights,train_dst_weights)
    h = model(masked_train_g, train_features)
    pos_score, neg_score = pred(h[stype][train_pos_edges[0]], h[dtype][train_pos_edges[1]]), pred(h[stype][train_neg_edges[0]], h[dtype][train_neg_edges[1]])

    pos_score = pos_score.to(device)
    neg_score = neg_score.to(device)
    loss = criterion(pos_score, neg_score)
    train_auc, train_acc = score(pos_score, neg_score)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.cpu().detach().numpy())
    

    model.eval()
    with torch.no_grad():
        val_neg_edges = weighted_neg_sampler(val_num_pos,1,val_src_weights,val_dst_weights)
        val_h = model(val_g, val_features)
        pos_score, neg_score = pred(val_h[stype][val_pos_edges[0]], val_h[dtype][val_pos_edges[1]]), pred(val_h[stype][val_neg_edges[0]], val_h[dtype][val_neg_edges[1]])
    val_loss = criterion(pos_score, neg_score)
    val_auc, val_acc = score(pos_score, neg_score)

    if val_auc > best_val_auc:
        best_epoch = epoch
        best_val_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")


    # print(
    #     "Epoch {:d} | Train Loss {:.4f} | Train AUC {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val AUC {:.4f} | Val Acc {:.4f} ".format(
    #         epoch + 1,
    #         loss.item(),
    #         train_auc,
    #         train_acc,
    #         val_loss.item(),
    #         val_auc,
    #         val_acc,
    #     )
    # )
    es(best_val_auc)
    if es.early_stop:
        break

print(
    "Best Epoch {:d} | Best valauc {:.4f} ".format(
        best_epoch + 1,
        best_val_auc,
    )
)






##### Test #####
print("-------------Test-------------")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

with torch.no_grad():
    val_neg_edges = weighted_neg_sampler(val_num_pos,1,val_src_weights,val_dst_weights)
    val_h = model(val_g, val_features)
    val_pos_score, val_neg_score = pred(val_h[stype][val_pos_edges[0]], val_h[dtype][val_pos_edges[1]]), pred(val_h[stype][val_neg_edges[0]], val_h[dtype][val_neg_edges[1]])
    test_neg_edges = weighted_neg_sampler(test_num_pos,1,test_src_weights,test_dst_weights)
    test_h = model(test_g, test_features)
    pos_score, neg_score = pred(test_h[stype][test_pos_edges[0]], test_h[dtype][test_pos_edges[1]]), pred(test_h[stype][test_neg_edges[0]], test_h[dtype][test_neg_edges[1]])


test_loss = criterion(pos_score, neg_score)
test_auc, auprc, f1, precision, recall, mcc = test_score(pos_score, neg_score, val_pos_score, val_neg_score)

print(
    "Test Loss {:.4f} | AUC {:.4f} | AUPRC {:.4f} | F1 {:.4f} | Precision {:.4f} | Recall {:.4f} | MCC {:.4f}".format(
        test_loss.item(),
        test_auc,
        auprc,
        f1,
        precision,
        recall,
        mcc
    )
)