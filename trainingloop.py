#imports
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, Sampler
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from copy import deepcopy
from numpy.random import randn 
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU,Tanh,LeakyReLU,ELU,SELU,GELU
from torch_geometric.nn import GINConv,EdgeConv, DynamicEdgeConv,global_add_pool, global_mean_pool, global_max_pool
import time
from tqdm import tqdm
from scipy.spatial import distance_matrix, Delaunay
import random
from torch_geometric.data import Data, DataLoader
import pickle
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold
from glob import glob
import os
import pandas as pd
import numpy as np
import pickle
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold, train_test_split
from random import shuffle
import math
from lifelines.utils import concordance_index as cindex

USE_CUDA = torch.cuda.is_available()
device = {True:'cuda:0',False:'cpu'}[USE_CUDA] 
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()
def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)
def toGeometricWW(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

class GNN(torch.nn.Module):

    def __init__(self, dim_features, dim_target, layers=[16,16,8],pooling='max',dropout = 0.0,conv='GINConv',gembed=False,**kwargs):
        """
        
        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.
        Raises
        ------
        NotImplementedError
            DESCRIPTION.
        Returns
        -------
        None.
        """
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim=layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {'max':global_max_pool,'mean':global_mean_pool,'add':global_add_pool}[pooling]
        self.gembed = gembed #if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim),GELU())
                self.linears.append(Sequential(Linear(out_emb_dim, dim_target),GELU()))
                
            else:
                input_emb_dim = self.embeddings_dim[layer-1]
                self.linears.append(Linear(out_emb_dim, dim_target))                
                if conv=='GINConv':
                    subnet = Sequential(Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim))
                    self.nns.append(subnet)
                    self.convs.append(GINConv(self.nns[-1], **kwargs))  # Eq. 4.2 eps=100, train_eps=False
                elif conv=='EdgeConv':
                    subnet = Sequential(Linear(2*input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim))
                    self.nns.append(subnet)                    
                    self.convs.append(EdgeConv(self.nns[-1],**kwargs))#DynamicEdgeConv#EdgeConv                aggr='mean'

                else:
                    raise NotImplementedError  
                    
        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)
def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

#Set up get pairs function -- Will have to optimise this later
def get_pairs(tuples):
  pairs_dataset = []
  for j in range(len(tuples)):
    if tuples[j][0] == 1:
      for i in range(len(tuples)):
        if i != j and tuples[i][1] > tuples[j][1]:
          pairs_dataset.append((i,j))
  shuffle(pairs_dataset)
  return pairs_dataset

def plot_curves(Nepochs, e_metric1, metric1, e_metric2, metric2):
    fig = plt.figure()
    ax0 = plt.add_subplot(121,metric1)
    ax1 = plt.add_subplot(121,metric2)
    ax0.plot(Nepochs,e_metric1)
    ax1.plot(Nepochs,e_metric2)
    plt.show()
    return None

# Set up const vals
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.01
EPOCHS = 10 # Total number of epochs
L1_WEIGHT = 0.001
SCHEDULER = None
BATCH_SIZE = 10

# This is set up to run on colab vvv
survival_file = 'drive/MyDrive/SlideGraph/NIHMS978596-supplement-1.xlsx'
cols2read = ['PFI', 'PFI.time']
TS = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID')  # path to clinical file
TS = TS[cols2read][TS.type == 'BRCA']
print(TS.shape)
#bdir = '/data/Immune_landscape/Slide_Graph_Resnet18/Graph_feats/'
#bdir = '/data/Immune_landscape/Slide_Graph_ALBRT/Graph_feats_1k/' #Graph_feats_1k/'
wsi_selected = 'drive/MyDrive/SlideGraph/slide_selection_final.txt'
filter = np.array(pd.read_csv(wsi_selected)).ravel()
Pids = np.array([p[:12] for p in filter])
# print(filter.shape); 
# import pdb; pdb.set_trace()
#'/data/Immune_landscape/Slide_Graph_ALBRT_Rank/Graph_feats_CC/' 
bdir = 'drive/MyDrive/SlideGraph/Graph_feats_CC_CPU/' # path to graphs

Exid = 'Slide_Graph CC_feats'
graphlist = glob(os.path.join(bdir, "*.pkl"))#[0:100]
GN = []
device = 'cuda:0'
cpu = torch.device('cpu')
dataset = []
pfi_and_times = []
for graph in tqdm(graphlist):
    G = pickleLoad(graph)
    G.to(cpu)
    TAG = os.path.split(graph)[-1].split('_')[0][:12]

    if TAG not in TS.index:
        continue
    if TAG not in Pids:
        continue

    status = TS.loc[TAG,:][1]
    pfi, pfi_time = TS.loc[TAG,:][0], TS.loc[TAG,:][1]
    pfi_and_times.append((pfi,pfi_time))
    G.y = toTensor([int(status)], dtype=torch.long, requires_grad = False)
    dataset.append(G)

for i,G in tqdm(enumerate(dataset)):
    W = radius_neighbors_graph(toNumpy(G.coords), 1500, mode='connectivity', include_self=False).toarray()
    g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
    g.coords = G.coords
    dataset[i]=g

print(len(dataset))
Y = np.array([float(G.y) for G in dataset])

# Split data and find paris for each 75% train 25% test
# Will look at validation later

num_train = math.floor(0.75 * len(dataset))
num_test = len(dataset) - num_train
train_dataset = dataset[:num_train]
test_dataset = dataset[-num_test:]
train_pairs = pfi_and_times[:num_train]
test_pairs = pfi_and_times[-num_test:]

if len(train_dataset) + len(test_dataset) != len(dataset):
  raise ValueError("Data not split correctly")
#Find pairs in each set
train_pairs_list = get_pairs(train_pairs)
test_pairs_list = get_pairs(test_pairs)

#Set up model and optimizer
model = GNN(dim_features=dataset[0].x.shape[1], dim_target = 1, layers = [16,16,8],
            dropout = 0.50, pooling = 'mean', conv = 'EdgeConv', aggr = 'max')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#training loop
epoch_loss = {}
epoch_acc = {}
epoch_loss['train'] = []
epoch_acc['train'] = []
model.train()
for epoch in tqdm(range(0,EPOCHS)):
    e_loss = 0
    e_acc = 0
    for i in range(0, len(train_pairs_list),BATCH_SIZE):
        optimizer.zero_grad()
        batch_pairs = train_pairs_list[i:i+BATCH_SIZE]
        temp = [idx for pair in batch_pairs for idx in pair]
        graph_set = list(set(temp))
        graphs = [train_dataset[i] for i in graph_set]
        '''
        data = [d for d in DataLoader([D[i] for i in bidx],batch_size = len(bidx))][0]
        data = data.to(device)
        '''
        batch_load = DataLoader(graphs,batch_size=len(graphs))
        #pdb.set_trace()
        for data in batch_load: #This for has only one iter.
          data = data.to(device)
        output,_,_ = model(data)
        #pdb.set_trace()
        loss = 0
        z = toTensor(0)
        num_pairs = len(batch_pairs)
        for (xi,xj) in batch_pairs:
            #the index of xi,xj in graph_set will give the index of relevant output
            graph_i, graph_j = graph_set.index(xi), graph_set.index(xj)
            #Compute loss function
            dz = output[graph_i] - output[graph_j]   
            #pdb.set_trace()
            loss += torch.max(z,1.0-dz)
        loss = loss/num_pairs
        '''
        #Perform L1 Reg. -- Remove for the moment
        norm = torch.cat([w_.view(-1) for w_ in model.parameters()])
        norm = torch.norm(norm, P)**P
        loss += L1_WEIGHT*norm
        '''
        #This can be changed at some other time
        loss.backward()
        optimizer.step()
        e_loss += loss.item()
        acc = 1 - loss.item()
        e_acc += acc
    epoch_loss['train'].append(e_loss)
    epoch_acc['train'].append(e_acc)

plot_curves(EPOCHS,epoch_loss['train'],'Loss',epoch_acc['train'],'Accuracy')

#Test evaluation on training data
#Need to alter these so it passes data in batches again and then appends outputs to list
#Also need to alter these so it takes the pfi_and_times pairs
model.eval()
loader = DataLoader(train_dataset,batch_size=len(train_dataset))
for data in loader: #This for has only one iter.
  data = data.to(device)
z,_,_ = model(data)
T = [train_dataset[i].PFI_TIME.cpu().numpy() for i in range(len(train_dataset))]
E = [train_dataset[i].PFI.cpu().numpy() for i in range(len(train_dataset))]
z = z.cpu().detach().numpy()
concord = cindex(T,z,E)
print(concord)

# Proper evaluation
model.eval()
load = DataLoader(test_dataset,batch_size=(len(test_dataset)))
for data in load:
  data = data.to(device)
z,_,_ = model(data)
T = [test_dataset[i].PFI_TIME.cpu().numpy() for i in range(len(test_dataset))]
E = [test_dataset[i].PFI.cpu().numpy() for i in range(len(test_dataset))]
z = z.cpu().detach().numpy()
concord = cindex(T,z,E)
print(concord)









