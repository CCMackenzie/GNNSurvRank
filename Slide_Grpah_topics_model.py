'''
Importing packages
'''

from re import L
import numpy as npds
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
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.model_selection import StratifiedKFold


"""
Created on Mon Oct 24 19:24:42 2016
This is a python implementation of Platt's normalization of classifier output scores to a probability value. It is an implementation of the Algorithm presented in:
Platt, John C. “Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.” In Advances in Large Margin Classifiers, 6174. MIT Press, 1999.
URL: 10s
Often times, after training a classifier, the output scores of a classifier need to be mapped to a more interpretable value. Platt's normalization is a classical method of doing just that. It fits a sigmoidal function z = 1/(1+exp(A*v+B) to the output scores v from the classifier and targets. The coefficients of the sigmoidal function can then be used to transform the output of any output from the classifier to a pseudo-probability value.
Implemented by: Dr. Fayyaz Minhas
@author: afsar
"""
class PlattScaling:
    def __init__(self):
        self.A = None
        self.B = None
    def fit(self,L,V):
        """
        Fit the sigmoid to the classifier scores V and labels L  using the Platt Mehtod
        Input:  V array-like of classifier output scores
                L array like of classifier labels (+1/-1 pr +1/0)
        Output: Coefficients A and B for the sigmoid function
        """
        def mylog(v):
            if v==0:
                return -200
            else: 
                return np.log(v)
        out = np.array(V)
        L = np.array(L)
        assert len(V)==len(L)
        target = L==1
        prior1 = np.float(np.sum(target))
        prior0 = len(target)-prior1    
        A = 0
        B = np.log((prior0+1)/(prior1+1))
        self.A,self.B = A,B
        hiTarget = (prior1+1)/(prior1+2)
        loTarget = 1/(prior0+2)
        labda = 1e-3
        olderr = 1e300
        pp = np.ones(out.shape)*(prior1+1)/(prior0+prior1+2)
        T = np.zeros(target.shape)
        for it in range(1,100):
            a = 0
            b = 0
            c 
            d = 0
            e = 0
            for i in range(len(out)):
                if target[i]:
                    t = hiTarget
                    T[i] = t
                else:
                    t = loTarget
                    T[i] = t
                d1 = pp[i]-t
                d2 = pp[i]*(1-pp[i])
                a+=out[i]*out[i]*d2
                b+=d2
                c+=out[i]*d2
                d+=out[i]*d1
                e+=d1
            if (abs(d)<1e-9 and abs(e)<1e-9):
                break
            oldA = A
            oldB = B
            err = 0
            count = 0
            while 1:
                det = (a+labda)*(b+labda)-c*c
                if det == 0:
                    labda *= 10
                    continue
                A = oldA+ ((b+labda)*d-c*e)/det
                B = oldB+ ((a+labda)*e-c*d)/det
                self.A,self.B = A,B
                err = 0
                for i in range(len(out)):            
                    p = self.transform(out[i])
                    pp[i]=p
                    t = T[i]
                    err-=t*mylog(p)+(1-t)*mylog(1-p)
                if err<olderr*(1+1e-7):
                    labda *= 0.1
                    break
                labda*=10
                if labda>1e6:
                    break
                diff = err-olderr
                scale = 0.5*(err+olderr+1)
                if diff>-1e-3*scale and diff <1e-7*scale:
                    count+=1
                else:
                    count = 0
                olderr = err
                if count == 3:
                    break
        self.A,self.B = A,B
        return self
    def transform(self,V):       
        return 1/(1+np.exp(V*self.A+self.B))
    
    def fit_transform(self,L,V):
        return self.fit(L,V).transform(V)

    def __repr__(self):
        A,B = self.A,self.B
        return "Platt Scaling: "+f'A: {A}, B: {B}'

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
    
def toGeometric(Gb,y,tt=1e-3):
    return Data(x=Gb.X, edge_index=(Gb.getW()>tt).nonzero().t().contiguous(),y=y)


class StratifiedSampler(Sampler):
    """Stratified Sampling
         return a stratified batch
    """
    def __init__(self, class_vector, batch_size = 10):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        """
        self.batch_size = batch_size
        self.n_splits = int(class_vector.size(0) / self.batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        skf = StratifiedKFold(n_splits= self.n_splits,shuffle=True)
        YY = self.class_vector.numpy()
        idx = np.arange(len(YY))
        return [tidx for _,tidx in skf.split(idx,YY)] #return array of arrays of indices in each batch

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def calc_roc_auc(target, prediction):
    return roc_auc_score(toNumpy(target),toNumpy(prediction[:,-1]))

def calc_pr(target, prediction):
    return average_precision_score(toNumpy(target), toNumpy(prediction[:, -1]))

#%% Graph Neural Network 
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
        
def decision_function(model,loader,device='cpu',outOnly=True,returnNumpy=False): 
    """
    generate prediction score for a given model
    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    loader : TYPE Dataset or dataloader
        DESCRIPTION.
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    outOnly : TYPE, optional 
        DESCRIPTION. The default is True. Only return the prediction scores.
    returnNumpy : TYPE, optional
        DESCRIPTION. The default is False. Return numpy array or ttensor
    Returns
    -------s
    Z : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    ZXn : TYPE
        DESCRIPTION. Empty unless outOnly is False
    """
    if type(loader) is not DataLoader: #if data is given
        loader = DataLoader(loader)
    if type(device)==type(''):
        device = torch.device(device)
    ZXn = []    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            output,zn,xn = model(data)
            if returnNumpy:
                zn,xn = toNumpy(zn),toNumpy(xn)
            if not outOnly:
                ZXn.append((zn,xn))
            if i == 0:
                Z = output
                Y = data.y
            else:
                Z = torch.cat((Z, output))
                Y = torch.cat((Y, data.y))
    if returnNumpy:
        Z,Y = toNumpy(Z),toNumpy(Y)
    
    return Z,Y,ZXn

def EnsembleDecisionScoring(Q,train_dataset,test_dataset,device='cpu',k=None):
    """
    Generate prediction scores from an ensemble of models 
    First scales all prediction scores to the same range and then bags them
    Parameters
    ----------
    Q : TYPE reverse deque or list or tuple
        DESCRIPTION.  containing models or output of train function
    train_dataset : TYPE dataset or dataloader 
        DESCRIPTION.
    test_dataset : TYPE dataset or dataloader 
        DESCRIPTION. shuffle must be false
    device : TYPE, optional
        DESCRIPTION. The default is 'cpu'.
    k : TYPE, optional
        DESCRIPTION. The default is None.
    Returns
    -------
    Z : Numpy array
        DESCRIPTION. Scores
    yy : Numpy array
        DESCRIPTION. Labels
    """
    
    Z = 0
    if k is None: k = len(Q)
    for i,mdl in enumerate(Q):            
        if type(mdl) in [tuple,list]:            mdl = mdl[0]
        zz,yy,_ = decision_function(mdl,train_dataset,device=device)            
        mdl.rescaler = PlattScaling().fit(toNumpy(yy),toNumpy(zz))
        zz,yy,_ = decision_function(mdl,test_dataset,device=device)
        zz,yy = mdl.rescaler.transform(toNumpy(zz)).ravel(),toNumpy(yy)
        Z+=zz
        if i+1==k: break
    Z=Z/k
    return  Z,yy
#%%   
class NetWrapper:
    def __init__(self, model, loss_function, device='cuda:0', classification=True):
        self.model = model
        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.classification = classification
    def _pair_train(self,train_loader,optimizer,clipping = None):
        """
        Performs pairwise comparisons with ranking loss
        """
        
        model = self.model.to(self.device)
        model.train()
        loss_all = 0
        acc_all = 0
        num_pairs = 0
        loss = 0
        assert self.classification

        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output,_,_ = model(data)
            pfi = data.PFI
            pfi_time = data.PFI_TIME
            num_pairs = 0
            z = toTensor([0])
            for j in range(len(pfi) - 1):
                if pfi[j] == 1:
                    for i in range(len(pfi) - 1):
                        if i != j and pfi_time[i] > pfi_time[j]:
                            num_pairs+=1
                            dfx = output[i,-1] - output[j,-1] #this is f(xi) - f(xj) in the loss fn
                            loss += torch.max(z, 1 - dfx)
            try:
                loss = loss/num_pairs
            except ZeroDivisionError:
                loss = 0
                continue
            #L1 regularization section
            L1_LAMBDA = 0.001 #Need to set this value at somepoint
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            total_loss = loss + (L1_LAMBDA*l1_norm)
            
            acc = loss
            total_loss.backward()
            #No idea what is going on here
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
    
    def classify_graphs(self, loader):
        Z,Y,_ = decision_function(self.model,loader,device=self.device)
        if not isinstance(Z, tuple):
            Z = (Z,)
        loss = 0
        auc_val = calc_roc_auc(Y, *Z)
        pr = calc_pr(Y, *Z)
        return auc_val, loss, pr
        
    def train(self, train_loader, max_epochs=100, optimizer=torch.optim.Adam, scheduler=None, clipping=None,
              validation_loader=None, test_loader=None, early_stopping=100, return_best = True, log_every=0):
        """
        
        Parameters
        ----------
        train_loader : TYPE
            Training data loader.
        max_epochs : TYPE, optional
            DESCRIPTION. The default is 100.
        optimizer : TYPE, optional
            DESCRIPTION. The default is torch.optim.Adam.
        scheduler : TYPE, optional
            DESCRIPTION. The default is None.
        clipping : TYPE, optional
            DESCRIPTION. The default is None.
        validation_loader : TYPE, optional
            DESCRIPTION. The default is None.
        test_loader : TYPE, optional
            DESCRIPTION. The default is None.
        early_stopping : TYPE, optional
            Patience  parameter. The default is 100.
        return_best : TYPE, optional
            Return the models that give best validation performance. The default is True.
        log_every : TYPE, optional
            DESCRIPTION. The default is 0.
        Returns
        -------
        Q : TYPE: (reversed) deque of tuples (model,val_acc,test_acc)
            DESCRIPTION. contains the last k models together with val and test acc
        train_loss : TYPE
            DESCRIPTION.
        train_acc : TYPE
            DESCRIPTION.
        val_loss : TYPE
            DESCRIPTION.
        val_acc : TYPE
            DESCRIPTION.
        test_loss : TYPE
            DESCRIPTION.
        test_acc : TYPE
            DESCRIPTION.
        """
        
        from collections import deque
        Q = deque(maxlen=10) # queue the last 10 models
        return_best = return_best and validation_loader is not None 
        val_loss, val_acc = -1, -1
        best_val_acc,test_acc_at_best_val_acc,val_pr_at_best_val_acc,test_pr_at_best_val_acc = -1,-1,-1,-1
        test_loss, test_acc = None, None
        time_per_epoch = []
        self.history = []   
        patience = early_stopping
        best_epoch = np.inf
        iterator = tqdm(range(1, max_epochs+1))        
        for epoch in iterator:
            updated = False

            if scheduler is not None:
                scheduler.step(epoch)
            start = time.time()
            
            train_acc, train_loss = self._pair_train(train_loader, optimizer, clipping)
            
            end = time.time() - start
            time_per_epoch.append(end)    

            if validation_loader is not None: 
                val_acc, val_loss, val_pr = self.classify_graphs(validation_loader)
            if test_loader is not None:
                test_acc, test_loss, test_pr = self.classify_graphs(test_loader)
            if val_acc>best_val_acc:
                best_val_acc = val_acc                
                test_acc_at_best_val_acc = test_acc
                val_pr_at_best_val_acc = val_pr
                test_pr_at_best_val_acc = test_pr
                best_epoch = epoch
                updated = True
                if return_best:
                    best_model = deepcopy(self.model)
                    Q.append((best_model,best_val_acc,test_acc_at_best_val_acc,val_pr_at_best_val_acc,test_pr_at_best_val_acc))

                if False:  
                    from vis import showGraphDataset,getVisData                
                    fig = showGraphDataset(getVisData(validation_loader,best_model,self.device,showNodeScore=False))
                    plt.savefig(f'./figout/{epoch}.jpg')
                    plt.close()
                    
            if not return_best:                   
                Q.append((deepcopy(self.model),val_acc,test_acc,val_pr,test_pr))
                   
            showresults = False
            if log_every==0: # show only if validation results improve
                showresults = updated
            elif (epoch-1) % log_every == 0:   
                showresults = True
                
            if showresults:                
                msg = f'Epoch: {epoch}, TR loss: {train_loss} TR perf: {train_acc}, VL perf: {val_acc} ' \
                    f'TE perf: {test_acc}, Best: VL perf: {best_val_acc} TE perf: {test_acc_at_best_val_acc} VL pr: {val_pr_at_best_val_acc} TE pr: {test_pr_at_best_val_acc}'
                tqdm.write('\n'+msg)                   
                self.history.append(train_loss)
                
            if epoch-best_epoch>patience: 
                iterator.close()
                break
            
        if return_best:
            val_acc = best_val_acc
            test_acc = test_acc_at_best_val_acc
            val_pr = val_pr_at_best_val_acc
            test_pr = test_pr_at_best_val_acc

        Q.reverse()    
        return Q,train_loss, train_acc, val_loss, np.round(val_acc, 2), test_loss, np.round(test_acc, 2), val_pr, test_pr


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

def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)

def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

def get_pairs(dataset):
    
    


if __name__ == '__main__':

    learning_rate = 0.01
    weight_decay = 0.01
    epochs = 300 # Total number of epochs
    split_fold = 5 # Stratified cross validation
    scheduler = None


    #topic_status_file =  'topic-50-binary.csv'
    #TS = pd.read_csv(topic_status_file).set_index('Patient ID')  # path to clinical file
    survival_file = 'drive/MyDrive/SlideGraph/NIHMS978596-supplement-1.xlsx'
    cols2read = ['ID','PFI', 'PFI.time']
    TS = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID') #Maybe put the set index bit back in here
    TS = TS[cols2read][TS.type == 'BRCA']
    print(TS.shape)
    #bdir = '/data/Immune_landscape/Slide_Graph_Resnet18/Graph_feats/'
    #bdir = '/data/Immune_landscape/Slide_Graph_ALBRT/Graph_feats_1k/' #Graph_feats_1k/'
    wsi_selected = '/data/Immune_landscape/Mutation_prediction/Mutation profile from H&E/slide_selection_final.txt'
    filter = np.array(pd.read_csv(wsi_selected)).ravel()
    Pids = np.array([p[:12] for p in filter])
    # print(filter.shape); 
    # import pdb; pdb.set_trace()
    #'/data/Immune_landscape/Slide_Graph_ALBRT_Rank/Graph_feats_CC/' 
    bdir = '/data/Immune_landscape/Slide_Graph_ALBRT/Graph_feats_CC/' # path to graphs

    Exid = 'Slide_Graph CC_feats'
    graphlist = glob(os.path.join(bdir, "*.pkl"))#[0:100]
    GN = []
    device = 'cuda:0'
    cpu = torch.device('cpu')
    dataset = []
    for graph in tqdm(graphlist):
        G = pickleLoad(graph)
        G.to(cpu)
        TAG =os.path.split(graph)[-1].split('_')[0][:12]
        if TAG not in TS.index:
            continue
        if TAG not in Pids:
            continue
        G.to(cpu)
        pfi, pfi_time = TS.loc[TAG,:][0], TS.loc[TAG,:][1]
        G.PFI = toTensor([int(pfi)], dtype=torch.int, requires_grad=False)
        G.PFI_TIME = toTensor([int(pfi_time)], dtype=torch.int, requires_grad=False)
        GN.append(G.x)
        dataset.append(G)

    # Normalise features
    GN = torch.cat(GN)
    Gmean, Gstd = torch.mean(GN, dim=0)+1e-10, torch.std(GN, dim=0)+1e-10
    for G in dataset:
       G.x = (G.x - Gmean) / Gstd

    print(len(dataset))
    Y = np.array([float(G.PFI) for G in dataset])

    #import pdb; pdb.set_trace()
    #print("Positive instances",sum(Y==1),' Negative instances ',sum(Y==0))
    skf = StratifiedKFold(n_splits=split_fold, shuffle=False)  # Stratified cross validation
    Vacc, Tacc, Vapr, Tapr, Test_ROC_overall, Test_PR_overall = [], [], [], [], [],  [] # Intialise outputs

    Fdata = []
    for trvi, test in skf.split(dataset, Y):
        train, valid = train_test_split(trvi, test_size=0.10, shuffle=True,
                                        stratify=np.array(Y)[trvi])  # 10% for validation and 90% for training 
        sampler = StratifiedSampler(class_vector=torch.from_numpy(np.array(Y)[train]), batch_size=10)
        train_dataset = [dataset[i] for i in train]
        tr_loader = DataLoader(train_dataset, batch_sampler=sampler)
        valid_dataset = [dataset[i] for i in valid]
        v_loader = DataLoader(valid_dataset, shuffle=False)
        test_dataset = [dataset[i] for i in test]
        tt_loader = DataLoader(test_dataset, shuffle=False)

        model = GNN(dim_features=dataset[0].x.shape[1], dim_target=1,
         layers=[16,16,8],dropout = 0.50,pooling='mean',conv='EdgeConv',aggr='max') #16,8,16

        #import pdb; pdb.set_trace()

        net = NetWrapper(model, loss_function=None, device=device)
        model = model.to(device=net.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_model, train_loss, train_acc, val_loss, val_acc, tt_loss, tt_acc, val_pr, test_pr = net.train(
            train_loader=tr_loader,
            max_epochs=epochs,
            optimizer=optimizer, 
            scheduler=scheduler,
            clipping=None,
            validation_loader=v_loader,
            test_loader=tt_loader,
            early_stopping=20,
            return_best=False,
            log_every=10)
        Fdata.append((best_model, test_dataset, valid_dataset))
        Vacc.append(val_acc)
        Tacc.append(tt_acc)
        Vapr.append(val_pr)
        Tapr.append(test_pr)
        print("\nfold complete", len(Vacc), train_acc, val_acc, tt_acc, val_pr, test_pr)

    # Averaged results of 5 folds
    print("avg Valid AUC=", np.mean(Vacc), "+/-", np.std(Vacc))
    print("avg Test AUC=", np.mean(Tacc), "+/-", np.std(Tacc))
    print("avg Valid PR=", np.mean(Vapr), "+/-", np.std(Vapr))
    print("avg Test PR=", np.mean(Tapr), "+/-", np.std(Tapr))

    # Use top 10 models in each fold and re-calculate the averaged results of 5 folds
    auroc = []
    aupr = []
    for idx in range(len(Fdata)):
        Q, test_dataset, valid_dataset = Fdata[idx]
        zz, yy = EnsembleDecisionScoring(Q, train_dataset, test_dataset, device=net.device, k=10)
        auroc.append(roc_auc_score(yy, zz))
        aupr.append(average_precision_score(yy, zz))

    print("avg Test AUC overall=", np.mean(auroc), "+/-", np.std(auroc))
    print("avg Test PR overall=", np.mean(aupr), "+/-", np.std(aupr))