#!/usr/bin/env python
# coding: utf-8

# In[40]:


from torch_geometric.datasets import MoleculeNet
import seaborn as sns
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[41]:


experiment = "rho2_40percent_Re50"

location = "../extrapolation/"+experiment+"/80_20_time_split/"

train_input = pd.read_csv(location+"train_input")
test_input = pd.read_csv(location+"test_input")

train_input_global = pd.read_csv(location+"train_input_scalar")
test_input_global = pd.read_csv(location+"test_input_scalar")

train_output = pd.read_csv(location+"train_output")
test_output = pd.read_csv(location+"test_output")

### Removing first columns ###
train_input = train_input.drop(['Unnamed: 0',"time"], axis=1)
train_input_global = train_input_global.drop('Unnamed: 0', axis=1)
train_output = train_output.drop('Unnamed: 0', axis=1)
    
test_input = test_input.drop(['Unnamed: 0',"time"], axis=1)
test_input_global = test_input_global.drop('Unnamed: 0', axis=1)
test_output = test_output.drop('Unnamed: 0', axis=1)

### drop any columns if needed ###
columns_to_drop = train_input.filter(regex='^(vpx_|vpy_|vpz_)').columns
train_input = train_input.drop(columns=columns_to_drop)

columns_to_drop = test_input.filter(regex='^(vpx_|vpy_|vpz_)').columns
test_input = test_input.drop(columns=columns_to_drop)

### reshaping to a nodal format ###
train_input = train_input.values.reshape(train_input.shape[0],16,4)
test_input = test_input.values.reshape(test_input.shape[0],16,4)

### global from dataframe to numpy ###
train_input_global = torch.tensor(train_input_global.values).float().clone().detach()
test_input_global = torch.tensor(test_input_global.values).float().clone().detach()

### outputs from dataframe to torch.tensor ###
train_output = train_output.values
test_output = test_output.values

### for rescaling back to original space ###
drag_min = pd.read_csv(location+"train_output_unscaled")["Drag"].values.min()
drag_max = pd.read_csv(location+"train_output_unscaled")["Drag"].values.max()


# In[42]:


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


# In[43]:


import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.dense import DenseGCNConv
from torch_geometric.nn import GATv2Conv

class GCN(torch.nn.Module):
    def __init__(self,embedding_size=200,batch_size=16,num_nodes=16,num_features=4):
        
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
#         self.linear_1 = Linear(1024+3,128)
        self.linear_1 = Linear(embedding_size+4,128)
        self.linear_2 = Linear(128,1)

    def forward(self, x, edge_index, x_scalar, batch_index):

        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.leaky_relu(hidden)
        
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        
        hidden = self.conv2(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        
        hidden = self.conv3(hidden, edge_index)
        hidden = F.leaky_relu(hidden)
        
        hidden = gap(hidden,batch_index)        
        hidden = torch.cat((hidden,x_scalar),axis=1)

        out = self.linear_1(hidden)
        out = F.leaky_relu(out)
        out = self.linear_2(out)

        return out

model = GCN().cuda()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


# In[44]:


# Wrap data in a data loader
NUM_GRAPHS_PER_BATCH = 16

train_loader = DataLoader(list(zip(train_combined,train_input_global)),
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

test_loader = DataLoader(list(zip(test_combined,test_input_global)),
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


# In[45]:


import os
from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Root mean squared error
#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.HuberLoss(reduction="mean",delta=0.25)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

### lr scheduler ###
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)

epoch_loss_train = []
epoch_loss_val = []
lr_list = []

# Find a unique save location
save_dir = "../extrapolation/"+experiment+"/80_20_time_split/"
trial_num = 1
while True:
    save_loc = os.path.join(save_dir, f'results_trial_{trial_num}')
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
        break
    trial_num += 1

best_model_path = os.path.join(save_loc, 'best_model.pt')

# Early stopping parameters
early_stopping_patience = 50
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(500):
    
    print(f'Starting Epoch {epoch+1}')
    model.train()
    train_loss = 0.0
    
    for batch,inputs_global in train_loader:
        
        batch = batch.to(device)
        inputs_global = inputs_global.float().to(device)
        
        optimizer.zero_grad()
        
        output = model(batch.x.float(), batch.edge_index, inputs_global, batch.batch)
        
        loss = loss_fn(output, batch.y)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_loader.dataset)
    epoch_loss_train.append(train_loss)
    
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch,inputs_global in test_loader:
            
            batch.to(device)
            inputs_global = inputs_global.float().to(device) 
            
            output = model(batch.x.float(), batch.edge_index, inputs_global, batch.batch)
            loss = loss_fn(output, batch.y)
            
            val_loss += loss.item() * batch.num_graphs
    
    val_loss /= len(test_loader.dataset)
    epoch_loss_val.append(val_loss)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    ### applying lr scheduling ###
    scheduler.step(val_loss)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Save intermediate models
    if epoch % 25 == 0:
        torch.save(model.state_dict(), os.path.join(save_loc, f'model_{epoch}.pt'))

    # Save losses
    np.save(os.path.join(save_loc, "epoch_loss_train.npy"), np.array(epoch_loss_train))
    np.save(os.path.join(save_loc, "epoch_loss_val.npy"), np.array(epoch_loss_val))

print("Training has completed")

