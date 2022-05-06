from sympy import Id
import torch
import pandas as pd
import numpy as np
import os
import tqdm


survival_file = 'drive/MyDrive/SlideGraph/NIHMS978596-supplement-1.xlsx'
cols2read = ['ID','PFI', 'PFI.time']
Survival = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'})
Survival = Survival[cols2read][Survival.type == 'BRCA'].reset_index(drop = True)

def get_pairs(df):
    pairs = dict()
    for id_j in range(df.shape[0]):
        if df.iloc[id_j]['PFI'] == 1:
            temp_id = df.iloc[id_j]['ID']
            temp_list = []
            for id_i in range(df.shape[0]):
                if df.iloc[id_i]['ID'] != temp_id and df.iloc[id_j]['PFI.time'] < df.iloc[id_i]['PFI.time']:
                    temp_list.append(df.iloc[id_i]['ID'])
                else:
                    continue
            pairs[temp_id] = temp_list
        else:
            continue

def get_pairs_numerical(df):
    pairs = dict()
    for id_j in range(df.shape[0]):
        if df.iloc[id_j]['PFI'] == 1:
            temp_id = df.iloc[id_j]['ID']
            temp_list = []
            for id_i in range(df.shape[0]):
                if df.iloc[id_i]['ID'] != temp_id and df.iloc[id_j]['PFI.time'] < df.iloc[id_i]['PFI.time']:
                    temp_list.append(id_i)
                else:
                    continue
            pairs[str(id_j)] = temp_list
        else:
            continue

USE_CUDA = True

def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))

def _pair_train(self,train_loader,optimizer,clipping = None):
    """
    Performs pairwise comparisons with ranking loss
    """
    model = self.model.to(self.device)
    model.train()
    loss_all = 0
    acc_all = 0
    assert self.classification
    for data in train_loader:
        
        data = data.to(self.device)
        
        optimizer.zero_grad()
        output,_,_ = model(data)
        y = data.y
        loss = 0
        no_pairs = 0
        z = toTensor([0])  
        for i in range(len(y)-1):
            for j in range(i+1,len(y)):
                if y[i]!=y[j]:
                    no_pairs+=1
                    dz = output[i,-1]-output[j,-1]
                    dy = y[i]-y[j]                        
                    loss+=torch.max(z, 1.0-dy*dz)
        loss=loss/no_pairs
        acc = loss
        loss.backward()

        try:
            num_graphs = data.num_graphs
        except TypeError:
            num_graphs = data.adj.size(0)

        loss_all += loss.item() * num_graphs
        acc_all += acc.item() * num_graphs
    

        if clipping is not None:  # Clip gradient before updating weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()

    return acc_all / len(train_loader.dataset), loss_all / len(train_loader.dataset)


for graph in tqdm(graphlist):
    G = pickleLoad(graph)
    G.to(cpu)
    TAG = os.path.split(graph)[-1].split('_')[0][:12]

    if TAG not in TS.index:
        continue
    if TAG not in Pids:
        continue
    G.to(cpu)
    status = TS.loc[TAG,:][1]
    G.y = toTensor([int(status)], dtype=torch.long, requires_grad=False) #I think I can get rid of this
    id = count
    G.ID = torch.tensor(id, dtype=torch.int)
    if str(id) in pairs_dict.keys():
        pair_list = pairs_dict[str(id)]
        G.pair_list = torch.tensor(pair_list, dtype=torch.int)
    GN.append(G.x)
    dataset.append(G)
    count += 1