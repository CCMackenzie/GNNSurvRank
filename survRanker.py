import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from plotit import *
USE_CUDA = torch.cuda.is_available() 
from torch.autograd import Variable
def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v
def toTensor(v,dtype = torch.float,requires_grad = False):       
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))
def toNumpy(v):
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()

print('Using CUDA:',USE_CUDA)


class SurvRanker:
    def __init__(self,lambdaw=0.01,p=1,Tmax = 200,lr=1e-1):#,lambdaw=0.010,p=1,Tmax = 100,lr=1e-1
        self.lambdaw = lambdaw
        self.p = p
        self.Tmax = Tmax
        self.lr = lr
        #return self
    def fit(self,X_train,T_train,E_train):        
        from sklearn.preprocessing import MinMaxScaler
        self.MMS = MinMaxScaler().fit(T_train.reshape(-1, 1))#rescale y-values
        T_train = 1e-3+self.MMS.transform(T_train.reshape(-1, 1)).flatten()        
        x = toTensor(X_train)
        y = toTensor(T_train)
        e = toTensor(E_train)
        N,D_in = x.shape
        H, D_out = D_in, 1
        model = torch.nn.Sequential(
            #torch.nn.Linear(D_in, H,bias=False),
            #torch.nn.Tanh(),   
            # torch.nn.Linear(H, H,bias=True),
            # torch.nn.Sigmoid(),  
            torch.nn.Linear(H, D_out,bias=False),
            torch.nn.Tanh()
        )
        #torch.nn.init.xavier_uniform_(model[0].weight)
        model=cuda(model)
        learning_rate = self.lr
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.0)
        TT = self.Tmax  
        lambdaw = self.lambdaw 
        p = self.p
        L = []
        dT = T_train[:, None] - T_train[None, :] #dT_ij = T_i-T_j
        dP = (dT>0)*E_train
        dP = toTensor(dP,requires_grad=False)>0 # P ={(i,j)|T_i>T_j ^ E_j=1}
        dY = (y.unsqueeze(1) - y)[dP]
        for t in (range(TT)):
            y_pred = model(x).flatten()
            dZ = (y_pred.unsqueeze(1) - y_pred)[dP]  
            loss = torch.mean(torch.max(toTensor([0],requires_grad=False),1.0-dZ)**p) #hinge loss
            
            # pwloss = torch.max(toTensor([0],requires_grad=False),1.0-dY*dZ)*dP#**p #hinge loss 
            # lsurv1 = torch.sum(pwloss,dim=1)/(1e-10+torch.sum(dP))
            # loss = torch.sum(lsurv1)
            
            #loss = torch.mean(1-torch.exp(dY[dP]*dZ[dP]))
            #import pdb; pdb.set_trace()
            
            #lsurv0 = torch.sum(pwloss,dim=0)/(1e-10+torch.sum(dP,dim=0))
            #lsurv = (lsurv1+lsurv0)/2.0
            #import pdb; pdb.set_trace()
            
            ww = torch.cat([w_.view(-1) for w_ in model[0].parameters()]) #weights of input
            loss+=lambdaw*torch.norm(ww, p)**p #regularize
            L.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
        ww = torch.cat([w_.view(-1) for w_ in model[0].parameters()])
        self.ww = ww
        self.L = L
        self.model = model
        return self
    def decision_function(self,x):
        x = toTensor(x)
        return toNumpy(self.model(x))
        