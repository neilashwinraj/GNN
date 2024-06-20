#!/usr/bin/env python
# coding: utf-8

#import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import clear_output
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch.nn as nn
import torch_geometric.nn as geom_nn
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

### Load Data (Ze Time series data) ###
location = "simple_connections_data/time_split/"
experiment = "rho10_20percent_Re50/"

train_input = np.load(location+experiment+"train_inputs.npy")
test_input = np.load(location+experiment+"test_inputs.npy")

### if not using local Reynolds numbers ###
#train_input = np.array(train_input[:,:,0:3])
#test_input = np.array(test_input[:,:,0:3])

train_inputs_global = torch.tensor(np.load(location+experiment+"train_input_scalar.npy"))
test_inputs_global = torch.tensor(np.load(location+experiment+"test_input_scalar.npy"))

train_output = np.load(location+experiment+"train_output.npy")
test_output = np.load(location+experiment+"test_output.npy")

### edge index for basic connections ###
edge_index = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])


train_combined = list()
test_combined = list()

### Stacking up train data ###
for i in range(len(train_input)):

    ### setting inputs ###
    x = torch.tensor(train_input[i]).float().clone().detach()
    
    ### adding drag force as y ###
    y = torch.tensor(train_output[i]).float().clone().detach()
    
    # all_data_graph_struct.append(Data(x=x , edge_index=torch.tensor(mirror_edge_index(edge_index)) , y=y))
    train_combined.append(Data(x=x.clone().detach() , edge_index=edge_index.clone().detach() , y=y[:,None].clone().detach()))

### Stacking up test data ###
for i in range(len(test_input)):

    ### setting inputs ###
    x = torch.tensor(test_input[i]).float().clone().detach()
    
    ### adding drag force as y ###
    y = torch.tensor(test_output[i]).float().clone().detach()
    
    # all_data_graph_struct.append(Data(x=x , edge_index=torch.tensor(mirror_edge_index(edge_index)) , y=y))
    test_combined.append(Data(x=x.clone().detach() , edge_index=edge_index.clone().detach() , y=y[:,None].clone().detach()))

import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.dense import DenseGCNConv

class GCN(torch.nn.Module):
    def __init__(self,embedding_size=256,batch_size=16,num_nodes=16,num_features=4):
        
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        # self.linear_1 = Linear(embedding_size*num_nodes,8)
        self.linear_1 = Linear(embedding_size+4,64)
        self.linear_2 = Linear(64,1)

    def forward(self, x, edge_index, x_scalar, batch_index):

        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.relu(hidden)
        
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.relu(hidden)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        
        hidden = self.conv3(hidden, edge_index)
        hidden = F.relu(hidden)

        ### Reshaping the tensors (for no aggregations) ###
        # hidden = hidden.reshape( batch.y.shape[0] , int((hidden.shape[0]/batch.y.shape[0])*hidden.shape[1]) )
        
        # Global Pooling (stack different aggregations)
        # hidden = torch.cat([gmp(hidden, batch_index), 
        #                     gap(hidden, batch_index)], dim=1)
        
        hidden = gap(hidden,batch_index)
        
        hidden = torch.cat((hidden,x_scalar),axis=1)
        out = self.linear_1(hidden)
        out = F.relu(out)
        out = self.linear_2(out)

        return out

model = GCN(num_features = train_input.shape[-1])
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

# Wrap data in a data loader
NUM_GRAPHS_PER_BATCH = 32

train_loader = DataLoader(list(zip(train_combined,train_inputs_global)), 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

test_loader = DataLoader(list(zip(test_combined,test_inputs_global)), 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

# Run Training

from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.000055)  

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

### lr scheduler ###
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=35, verbose=True)

epoch_loss_train = list()
epoch_loss_val = list()
lr_list = list()

save_loc = location+experiment

for epoch in range(0,500):
    print(f'Starting Epoch {epoch+1}')

    current_loss = 0.0
    loss_train = list()
    loss_val = list()
    
    for batch,inputs_global in train_loader:

        batch.to(device)
        inputs_global = inputs_global.float().cuda() 

        optimizer.zero_grad()

        pred = model( batch.x.float() , batch.edge_index, inputs_global, batch.batch)
        
        loss = loss_fn(pred, batch.y)
        loss.backward()  
        
        # Update using the gradients
        optimizer.step()   

        current_loss += loss.item()
        loss_train.append(loss.item())
        
    for batch,inputs_global in test_loader:

        batch.to(device)
        inputs_global = inputs_global.float().cuda() 
        
        pred = model( batch.x.float() , batch.edge_index,inputs_global, batch.batch)
        
        loss = loss_fn(pred,batch.y)
        loss_val.append(loss.item())
        
    print(f'Epoch {epoch+1} finished with training loss = '+str(np.array(loss_train).mean()))
    print(f'testing loss = '+str(np.array(loss_val).mean()) + '\n' )

    epoch_loss_train.append(np.array(loss_train).mean())
    epoch_loss_val.append(np.array(loss_val).mean())

    ### applying lr scheduling ###
    lr_list.append(optimizer.param_groups[0]['lr'])
    scheduler.step(epoch_loss_train[-1])

    if epoch%25==0:
        
        torch.save(model.state_dict(), save_loc+'model_'+str(epoch))

    np.save(save_loc+"epoch_loss_train",epoch_loss_train)
    np.save(save_loc+"epoch_loss_val",epoch_loss_val)

print("Training has completed")


# In[16]:


import matplotlib.pyplot as plt
plt.semilogy(epoch_loss_train)
plt.semilogy(epoch_loss_val)
plt.semilogy(lr_list)
plt.savefig(location+experiment+"train_val_loss_vs_epochs")
print(epoch_loss_train[-1],epoch_loss_val[-1])

