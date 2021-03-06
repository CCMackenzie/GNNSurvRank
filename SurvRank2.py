#imports
from xmlrpc.client import Boolean
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, Sampler
import torch.utils.data as data_utils
from Newtraining import BATCH_SIZE, loss_fn
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
from statistics import mean
from lifelines.utils import concordance_index as cindex
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

# Misc. functions
USE_CUDA = torch.cuda.is_available()
device = {True:'cuda:0',False:'cpu'}[USE_CUDA] 
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = True):
    return torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad)

def toTensorGPU(v,dtype = torch.float,requires_grad = True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))

def toNumpy(v):
    if type(v) is not torch.Tensor: return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)
    
def toGeometric(Gb,y,tt=1e-3):
    return Data(x=Gb.x, edge_index=(Gb.get(W)>tt).nonzero().t().contiguous(),y=y)

def toGeometricWW(X,W,y,tt=0):    
    return Data(x=toTensor(X,requires_grad = False), edge_index=(toTensor(W,requires_grad = False)>tt).nonzero().t().contiguous(),y=toTensor([y],dtype=torch.long,requires_grad = False))

def graph_load(batch, directory = "Graphs"):
  return [torch.load(directory + '/' + graph + '.g') for graph in batch]

class GNN(torch.nn.Module):
    def __init__(self, dim_features, dim_target, layers=[16,16,8],pooling='max',dropout = 0.0,conv='GINConv',gembed=False,**kwargs):
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
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        #import pdb; pdb.set_trace()
        
        out = 0
        pooling = self.pooling
        Z = 0
        import torch.nn.functional as F
        for layer in range(self.no_layers):            
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z+=z
                dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer-1](x,edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z+=z
                    dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(self.linears[layer](pooling(x, batch)), p=self.dropout, training=self.training)
                out += dout

        return out,Z,x

def get_predictions(model, keys):
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(0,len(keys), BATCH_SIZE):
            graphs = graph_load(keys[i:i+BATCH_SIZE])
            loader = DataLoader(graphs, batch_size=BATCH_SIZE)
            for data in loader:
                data = data.to(device)
            z,_,_ = model(data)
            z = z.cpu().detach().numpy()
            for j in range(len(z)):
                outputs.append(z[j][0])
    return outputs

class NetWrapper:
    def __init__(self,model,loss_function, device='cuda:0',mode='Survival',batch_size = 10) -> None:
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.mode = mode
        self.batch_size = batch_size
    
    def loss_fn(self,batch,testing=False) -> float:
        model = self.model.to(self.device)
        z = toTensorGPU(0)
        loss = 0
        unzipped = [j for pair in batch for j in pair]
        graph_set = list(set(unzipped))
        graphs = graph_load(graph_set)
        batch_load = DataLoader(graphs, batch_size = len(graphs))
        for data in batch_load:
            data = data.to(device)
        if testing:
            model.eval()
        else:
            model.train()
        with torch.set_grad_enabled(testing):
            output,_,_ = model(data)
        num_pairs = len(batch)
        for (xi,xj) in batch:
            graph_i, graph_j = graph_set.index(xi), graph_set.index(xj)
            # Compute loss function
            dz = output[graph_i] - output[graph_j]
            loss += torch.max(z, 1.0 - dz)
        if testing:
            return loss.item(), num_pairs
        loss = loss/num_pairs
        return loss

    def validation_loss_disk(self,pairs_list) -> float:
        tot_pairs = 0
        tot_loss = 0
        batch_size = self.batch_size
        for j in range(0, len(pairs_list), batch_size):
            b_pairs = pairs_list[j:j+batch_size]
            loss, pairs = self.loss_fn(b_pairs,testing=True)
            tot_pairs += pairs
            tot_loss += loss
        return tot_loss/tot_pairs

    def concord(self,dataset) -> float:
        keys = [key for key in dataset]
        outputs = get_predictions(self.model,keys)
        T = [dataset[key][1] for key in keys]
        E = [dataset[key][0] for key in keys]
        return cindex(T,outputs,E)

    def convergence_curves(self,batches,e_metric):
        '''
        NEED TO FIX HERE
        '''
        Nepochs = np.arange(batches)
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss', color=color)
        ax1.plot(Nepochs, e_metric['train'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx() 
        color = 'tab:blue'
        ax2.set_ylabel('Test Loss', color=color)
        ax2.plot(Nepochs, e_metric['test'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

    def train(self,max_batches=10_000,optimizer=torch.optim.Adam,early_stopping=100,return_best=False,
            num_logs=50, training_data = None, validation_data = None, batch_size = 10,convergence=True,
            output = True):
        model = self.model.to(self.device)
        counter = 0
        best_c_val = 0
        not_improved = 0
        if validation_data:
            VALIDATION = True
        print("Number of batches used for training "+ str(max_batches))
        log_iterval = max_batches // num_logs
        loss_vals = {}
        loss_vals['train'] = []
        loss_vals['test'] = []
        z = toTensorGPU(0)
        for i in tqdm(range(max_batches)):
            if counter < len(training_data):
                optimizer.zero_grad()
                # Get a batch of pairs
                batch_pairs = training_data[counter:counter+batch_size]
                loss = self.loss_fn(batch_pairs)
                lv = loss.item()
                loss.backward()
                optimizer.step()
                counter += batch_size
            # This is to resolve list index errors with large NUM_BATCHES vals
            if counter >= len(training_data):
                counter = 0
                # Could shuffle here if there are examples that are never considered
            if i % log_iterval == 0:
                if VALIDATION:
                    val_loss = self.validation_loss_disk(validation_data)
                    loss_vals['test'].append(val_loss)
                loss_vals['train'].append(lv) # This might not work
                if output:
                    print("Current Loss Val: " + str(lv) + "\n")
                    print("Current Vali Loss Val: " + str(val_loss) + "\n")
            if i%10 == 0: # This needs fixing
                c_val = self.concord()
                if c_val > best_c_val:
                    best_c_val = c_val
                    # Would implement a save best model here as well
                else:
                    not_improved += 1
            if not_improved == 2:
                train_len = i
                break
        if not train_len:
            train_len = max_batches
        if convergence:
            # Show convergence curves
            self.convergence_curves(train_len,loss_vals)

class Evaluator:
    def __init__(self,model,device = 'cuda:0',batchsize = 10):
        self.model = model
        self.device = torch.device(device)
        self.batchsize = batchsize

    def concordance(self, dataset):
        keys = [key for key in dataset]
        outputs = get_predictions(self.model,keys)
        T = [dataset[key][1] for key in keys]
        E = [dataset[key][0] for key in keys]
        return cindex(T,outputs,E)
    
    def kaplan_meier_est(self,dataset, split_val, mode = 'Train'):
        keys = [key for key in dataset]
        outputs = get_predictions(self.model, keys)
        T = [dataset[key][1] for key in keys]
        E = [dataset[key][0] for key in keys]
        mid = np.median(outputs)
        if mode != 'Train':
            if split_val > 0:
                mid = split_val
        else:
            print(mid) # Might change this
        T_hi = []
        T_lo = []
        E_hi = []
        E_lo = []
        for i in range(len(outputs)):
            if outputs <= mid: # Swapped from >= to <=
                T_hi.append(T[i])
                E_hi.append(E[i])
            else:
                T_lo.append(T[i])
                E_lo.append(E[i])
        km_hi = KaplanMeierFitter()
        km_lo = KaplanMeierFitter()
        ax = plt.subplot(111)
        ax = km_hi.fit(T_hi, event_observed=E_hi, label = 'High').plot_survival_function(ax=ax)
        ax = km_lo.fit(T_lo, event_observed=E_lo, label = 'Low').plot_survival_function(ax=ax)
        add_at_risk_counts(km_hi, km_lo, ax=ax)
        plt.title('Kaplan-Meier estimate')
        plt.ylabel('Survival probability')
        plt.show()
        plt.tight_layout()
        results = logrank_test(T_lo, T_hi, E_lo, E_hi)
        print("p-value %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))

def output_and_loss(model,batch,testing = False, c_index = False): # Not completed will continue to look at this
    z = toTensorGPU(0)
    loss = 0
    unzipped = [j for pair in batch for j in pair]
    graph_set = list(set(unzipped))
    graphs = graph_load(graph_set)
    batch_load = DataLoader(graphs, batch_size = len(graphs))
    for data in batch_load:
        data = data.to(device)
    if testing:
        model.eval()
    else:
        model.train()
    with torch.set_grad_enabled(testing):
        output,_,_ = model(data)
    num_pairs = len(batch)
    for (xi,xj) in batch:
        graph_i, graph_j = graph_set.index(xi), graph_set.index(xj)
        # Compute loss function
        dz = output[graph_i] - output[graph_j]
        loss += torch.max(z, 1.0 - dz)
    if testing and not c_index:
        return loss.item(), num_pairs
    loss = loss/num_pairs
    return loss
