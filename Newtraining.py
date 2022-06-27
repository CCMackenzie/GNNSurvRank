'''
This code has been writen to run on google colab with a mounted google drive for data
This implementation uses on disk storage for the graphs to get around large data using up systme memory
'''
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
import pdb
from statistics import mean

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

#This needs changing so it takes two axis I think
def plot_curves(Nepochs, e_metric, metric1):
    Nepochs = np.arange(Nepochs)
    plt.plot(Nepochs,e_metric['train'],label = "Training")
    plt.plot(Nepochs,e_metric['test'], label = "Test")
    plt.legend()
    plt.show()
    return None

def new_plot(Nepochs, e_metric):
    Nepochs = np.arange(Nepochs)

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
import math
from random import shuffle
from itertools import islice

def get_pairs(tuples): # Need to change this func to work with saved data
  pairs_dataset = []
  for j in range(len(tuples)):
    if tuples[j][0] == 1:
      for i in range(len(tuples)):
        if i != j and tuples[i][1] > tuples[j][1]:
          pairs_dataset.append((i,j))
  shuffle(pairs_dataset)
  return pairs_dataset

def new_pair_find(data):
  pairs_dataset = []
  for key_j in data:
    event_j, time_j = data[key_j][0], data[key_j][1]
    if event_j == 1:
      for key_i in data:
        event_i, time_i = data[key_i][0], data[key_i][1]
        if key_j != key_i and time_i > time_j:
          pairs_dataset.append((key_i,key_j))
  shuffle(pairs_dataset)
  return pairs_dataset

def loss_fn(model,batch,testing = False):
    z = toTensorGPU(0)
    loss = 0
    unzipped = [j for pair in batch for j in pair]
    graph_set = list(set(unzipped))
    batch_load = DataLoader(graph_set, batch_size = len(graph_set))
    for data in batch_load:
        data = data.to(device)
    if testing:
        model.eval()
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

def validation_loss_disk(val_data, pairs_list):
    tot_pairs = 0
    tot_loss = 0
    for j in range(0, len(pairs_list), BATCH_SIZE):
        b_pairs = pairs_list[j:j+BATCH_SIZE]
        loss, pairs = loss_fn(model,b_pairs,testing = True)
        tot_pairs += pairs
        tot_loss += loss
    epoch_val_loss = tot_loss / tot_pairs
    return epoch_val_loss

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

def C_index_eval(dataset, model):
    keys = [key for key in dataset]
    outputs = get_predictions(model,keys)
    T = [dataset[key][1] for key in keys]
    E = [dataset[key][0] for key in keys]
    concord = cindex(T,outputs,E)
    return concord

def concord_plot(c_vals):
    plt.plot(c_vals,linestyle='dotted')
    plt.show()

# Set up const vals
LEARNING_RATE = 0.01  
WEIGHT_DECAY = 0.01
EPOCHS = 5 # Total number of epochs
L1_WEIGHT = 0.001
SCHEDULER = None
BATCH_SIZE = 10 #Reduce further and allow larger files
NUM_BATCHES = 10_000
NUM_LOGS = 50 # How many times in the training loss is stored
P = 1
SHUFFLE_NET = True #Select what feature set to use
MAX_FILESIZE = 100_000_000_000_000 #Set as large number will remove file restrictions later
VALIDATION = True
NORMALIZE = False
CENSORING = True
FRAC_TRAIN = 0.8
CONCORD_TRACK = True

#from re import X
import pandas as pd
# This is set up to run on colab vvv
#Confirm that features file corresponds to xlsx file 
survival_file = 'drive/MyDrive/SlideGraph/NIHMS978596-supplement-1.xlsx'
cols2read = ['OS','OS.time'] #['PFI', 'PFI.time'] 
TS = pd.read_excel(survival_file).rename(columns= {'bcr_patient_barcode':'ID'}).set_index('ID')  # path to clinical file
TS = TS[cols2read][TS.type == 'BRCA']
print(TS.shape)
#wsi_selected = 'drive/MyDrive/SlideGraph/slide_selection_final.txt'
#filter = np.array(pd.read_csv(wsi_selected)).ravel()
#Pids = np.array([p[:12] for p in filter])
bdir = 'drive/MyDrive/SlideGraph/Graph_feats_CC_CPU/' # path to graphs
if SHUFFLE_NET:
  bdir = 'drive/MyDrive/SlideGraph/graphs_featuresresnet_cluster_08_1111ERslides/'
Exid = 'Slide_Graph CC_feats'
graphlist = glob(os.path.join(bdir, "*.pkl"))#[0:100]
print(len(graphlist))
device = 'cuda:0'
cpu = torch.device('cpu')
dataset = []
# Set up directory for on disk dataset
directory = 'Graphs'
try: # This could be made better
  os.mkdir(directory)
except FileExistsError:
  pass
event_and_times = {}
for graph in tqdm(graphlist):
    if SHUFFLE_NET:
      filesize = os.path.getsize(graph)
      if filesize < MAX_FILESIZE: # Remove this now
        G = torch.load(graph, map_location='cpu') # Seem to get an error here randomly "Transport endpoint is not connected"
        G = Data(**G.__dict__)
      else:
        continue
    else:
      G = pickleLoad(graph)
      G.to('cpu')
    TAG = os.path.split(graph)[-1].split('_')[0][:12]
    #if TAG not in TS.index:
        #continue
    #if TAG not in Pids:
        #continue
    status = TS.loc[TAG,:][1]
    event, event_time = TS.loc[TAG,:][0], TS.loc[TAG,:][1]
    try:
      G.y = toTensorGPU([int(status)], dtype=torch.long, requires_grad = False)
    except ValueError:
      continue
    W = radius_neighbors_graph(toNumpy(G.coords), 1500, mode="connectivity",include_self=False).toarray()
    g = toGeometricWW(toNumpy(G.x),W,toNumpy(G.y))
    g.coords = G.coords
    g.event = toTensor(event)
    g.e_time = toTensor(event_time)
    event_and_times[TAG] = (event,event_time)
    #dataset.append(G)
    torch.save(g,'Graphs/'+TAG+'.g') #Need to save to appropriate location
  
if NORMALIZE: #Need to make this an online mean and stdev calc.
  GN = []
  for graph in dataset:
    GN.append(G.x)
  GN = torch.cat(GN)
  Gmean, Gstd = torch.mean(GN,dim=0)+1e-10, torch.std(GN,dim=0)+1e-10
  GN = None #Free up memory
  for G in dataset:
    G.x = (G.x - Gmean)/Gstd

def censor_data(data: dict, censor_time: int): # The censor time here is measured in years
    # Convert the time to days for the purpose of the data
    cen_time = 365 * censor_time
    for key in data:
        event, time = data[key][0], data[key][1]
        if time > cen_time:
            data[key] = (0, cen_time)
        else:
          continue 
    return data

# Small function to split the dictionaries:
# Takes big dictionary and returns test and training split
def dict_split(data: dict, directory: str, train: float, test: float):
    num_graphs = len(os.listdir(directory))
    num_train = math.floor(0.9 * len(event_and_times))
    num_test = len(event_and_times) - num_train
    train_data = dict(islice(data.items(), 0, num_train))
    test_data = dict(islice(data.items(), num_train, num_graphs))
    if len(train_data) + len(test_data) != num_graphs:
      raise ValueError("Invalid Dictionary split")
    print('Number of training graphs: ' + str(len(train_data)))
    print('Number of test graphs: ' + str(len(test_data)))
    print('Number of test/validation graphs: ' + str(len(test_data)))
    return train_data, test_data

def graph_load(batch):
    return [torch.load(directory + '/' + graph + '.g') for graph in batch]


num_graphs = len(os.listdir(directory))
print(num_graphs)
print(len(event_and_times))
#Y = np.array([float(G.y) for G in dataset])
# Split data and find paris for each 75% train 25% test
# Will look at validation later

if num_graphs != len(event_and_times):
    raise ValueError("Inconsistent data values")

#Ensure the pfi_dataset and the graphs are aligned
num_train = math.floor(0.75 * len(event_and_times))
num_test = len(event_and_times) - num_train

if CENSORING:
    event_and_times = censor_data(event_and_times)

train_dataset, test_dataset = dict_split(event_and_times, 'Graphs', 0.75, 0.25)
# Run pair finding over the given dictionaries
train_pairs_list = new_pair_find(train_dataset)
test_pairs_list = new_pair_find(test_dataset)

print('Total number of examples: ' + str(len(event_and_times)))
print('Number of training examples: ' + str(len(train_dataset)))
print('Number of test examples: ' +str(len(test_dataset)))
print('Number of valid pairs for training: ' + str(len(train_pairs_list)))
print('Number of valid pairs for testing: ' + str(len(test_pairs_list)))

#Load in a graph for the model parameters
graph = torch.load(directory + '/TCGA-3C-AALI.g')

#Set up model and optimizer
model = GNN(dim_features=graph.x.shape[1], dim_target = 1, layers = [32,16,8],
            dropout = 0.0, pooling = 'mean', conv='GINConv', aggr = 'max')
# Have changed dataset[0].x.shpae[1] to graph.x.shape[1]
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training Loop
counter = 0
b_loss = 0
print("Number of batches used for training "+ str(NUM_BATCHES))
log_iterval = NUM_BATCHES // NUM_LOGS
loss_vals = {}
loss_vals['train'] = []
loss_vals['test'] = []
concords = []
# Running list of previous losses to get more accurate training loss at intervals
prev_losses = [0] * 50 # Queue of previous 50 batches loss values for averaging
# Not sure about the positioning of this but this seems ok
model.train()
for i in tqdm(range(NUM_BATCHES)):
    prev_losses.pop(0)
    if counter < len(train_pairs_list):
        optimizer.zero_grad()
        # Get a batch of pairs
        batch_pairs = train_pairs_list[counter:counter+BATCH_SIZE]
        loss = loss_fn(model,batch_pairs)
        prev_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        counter += BATCH_SIZE
    # This is to resolve list index errors with large NUM_BATCHES vals
    else:
        counter = 0
        # Could shuffle here if there are examples that are never considered
    if i % log_iterval == 0:
        if VALIDATION:
            val_loss = validation_loss_disk(test_dataset, test_pairs_list)
            loss_vals['test'].append(val_loss)
            if CONCORD_TRACK:
                c_val = C_index_eval(test_dataset,model)
                concords.append(c_val)
        if len(prev_losses) != 50:
            raise(ValueError)
        lv = mean(prev_losses)
        loss_vals['train'].append(lv) # This might not work
        print("Current Loss Val: " + str(lv) + "\n")
        print("Current Vali Loss Val: " + str(val_loss) + "\n")
if CONCORD_TRACK:
    concord_plot()

def K_M_Curves(dataset, model, split_val, mode = 'Train'):
    keys = [key for key in dataset]
    outputs = get_predictions(model,keys)
    T = [dataset[key][1] for key in keys]
    E = [dataset[key][0] for key in keys]
    mid = np.median(outputs)
    if mode != 'Train':
        if split_val > 0:
            mid = split_val
    else:
        print(mid)
    T_high = []
    T_low = []
    E_high = [] 
    E_low = []
    for i in range(len(outputs)):
      if outputs[i] <= mid:
        T_high.append(T[i])
        E_high.append(E[i])
      else:
        T_low.append(T[i])
        E_low.append(E[i])
    km_high = KaplanMeierFitter()
    km_low = KaplanMeierFitter()
    ax = plt.subplot(111)
    ax = km_high.fit(T_high, event_observed=E_high, label = 'High').plot_survival_function(ax=ax)
    ax = km_low.fit(T_low, event_observed=E_low, label = 'Low').plot_survival_function(ax=ax)
    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(km_high, km_low, ax=ax)
    plt.title('Kaplan-Meier estimate')
    plt.ylabel('Survival probability')
    plt.show()
    plt.tight_layout()
    from lifelines.statistics import logrank_test
    results = logrank_test(T_low, T_high, E_low, E_high)
    print("p-value %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))