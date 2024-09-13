#!/usr/bin/env python
# coding: utf-8

# # 5 history with drag seq2seq

# In[1]:


# from torch_geometric.datasets import MoleculeNet
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import clear_output
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch.nn as nn
import torch_geometric.nn as geom_nn
import copy
from torch.optim.lr_scheduler import  ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import pickle
import math

import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.dense import DenseGCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[2]:


### Load the list from the pickle file ###
### Train Data ###
loc = "../simple_connections_data/temporal_split/"
experiment = "rho2_20percent_Re50/history_5/"

save_loc = loc+experiment+"results_history_5_seq2seq/"

### Train Inputs ###
with open(loc+experiment+'train_data_np_scaled.pkl', 'rb') as f:
    train_data_np_scaled = pickle.load(f)

### Test Inputs ###
with open(loc+experiment+'test_data_np_scaled.pkl', 'rb') as f:
    test_data_np_scaled = pickle.load(f)
        
### Train Outputs ###
train_output = np.load(loc+experiment+"train_output_scaled.npy")

### Test Outputs ###
test_output = np.load(loc+experiment+"test_output_scaled.npy")


# # Combined_model

# In[95]:


def make_batch_index_tensor(node_length, batch_size):
    # Create a tensor containing values from 0 to batch_size - 1
    values = torch.arange(batch_size)
    # Repeat each value node_length times and flatten the result
    repeated_values = values.repeat_interleave(node_length)
    return repeated_values


# In[96]:


def make_edge_index_tensor(batch_size):
    
    edge_index = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                               [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
    
    if batch_size==1:
    
        return edge_index
    
    if batch_size>1:
        
        add = torch.cat([ torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]) + 16*(i+1) for i in range(batch_size-1)],axis=1)
        
        edge_index_expanded = torch.cat((edge_index,add),axis=1)
    
        return edge_index_expanded


# In[97]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool as gap

class GNN_with_attention(nn.Module):
    
    def __init__(self, num_gcn_layers=2, gcn_embedding_size=64, 
                 num_nodes=16, num_features=4, dropout_prob=0.20, heads=2):
        
        super(GNN_with_attention, self).__init__()
        torch.manual_seed(42)

        self.num_gcn_layers = num_gcn_layers
        self.gcn_embedding_size = gcn_embedding_size
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.heads = heads
        self.dropout_prob = dropout_prob

        self.initial_conv = GATv2Conv(self.num_features, self.gcn_embedding_size, self.heads)
        
        self.convs = nn.ModuleList([GATv2Conv(self.gcn_embedding_size * self.heads, self.gcn_embedding_size, self.heads) 
                                    for _ in range(self.num_gcn_layers - 1)])

        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x, edge_index, x_scalar, batch_index):
      
        hidden = F.leaky_relu(self.initial_conv(x, edge_index))
      
        for i, conv in enumerate(self.convs):
            hidden = F.leaky_relu(conv(hidden, edge_index))
            if i % 2 == 1:
                hidden = self.dropout(hidden)

        hidden = gap(hidden, batch_index)
        hidden = torch.cat((hidden, x_scalar), axis=1)

        return hidden

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesTransformerEncoder(nn.Module):
    
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, max_len):
        super(TimeSeriesTransformerEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead
                                                  ,dim_feedforward=self.dim_feedforward)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Linear(self.d_model, 6)

    def forward(self, x):

        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.pooling(x)  # (batch_size, d_model, 1)
        x = x.squeeze(2)  # (batch_size, d_model)
        x = self.output_projection(x)  # (batch_size, 1)
        
        return x

class GNN_Transformer(nn.Module):
    
    def __init__(self, num_gcn_layers=2, gcn_embedding_size=128, 
                 num_nodes=16, num_features=4, dropout_prob=0.2, heads=2,  ### GAT parameters
                 input_dim=260, d_model=124, nhead=4, num_layers=2, dim_feedforward=64, max_len=5):  ### Transformer parameters
        super(GNN_Transformer, self).__init__()
        
        self.num_gcn_layers = num_gcn_layers 
        self.gcn_embedding_size = gcn_embedding_size 
        self.num_nodes=num_nodes
        self.num_features=num_features 
        self.dropout_prob=dropout_prob 
        self.heads=heads  ### GAT parameters
        
        self.input_dim=input_dim 
        self.d_model=d_model 
        self.nhead=nhead 
        self.num_layers=num_layers 
        self.dim_feedforward=dim_feedforward 
        self.max_len=max_len ### Transformer parameters
        
        self.GNN = GNN_with_attention(self.num_gcn_layers,  self.gcn_embedding_size,
                                       self.num_nodes, self.num_features, 
                                      self.dropout_prob, self.heads)
        
        self.transformer = TimeSeriesTransformerEncoder(self.input_dim, self.d_model, 
                                                        self.nhead,self.num_layers, self.dim_feedforward, max_len)

    def forward(self, x, x_scalar):
        
        # x shape: (batch_size, seq_len, num_nodes, num_features) # try combinignthe seq_len and num_features 
        # edge_index, x_scalar, and batch_index should be provided for each graph in the sequence
        
        batch_size, seq_len, num_nodes, num_features = x.size()
        gnn_embeddings = []
        
        for i in range(seq_len):
            
            x_t = x[:, i, :, :].reshape(-1,num_features)  # shape: (batch_size * num_nodes, num_features)
            
            edge_index_t = make_edge_index_tensor(batch_size).cuda() # Assuming edge_index is the same for all graphs in the sequence

            x_scalar_t = x_scalar[:, i, :]  # shape: (batch_size, scalar_features)

            batch_index_t = make_batch_index_tensor(num_nodes,batch_size).cuda()  # shape: (batch_size * num_nodes,)

            gnn_embedding_t = self.GNN(x_t, edge_index_t, x_scalar_t, batch_index_t)
            gnn_embeddings.append(gnn_embedding_t.unsqueeze(1))  # Add sequence dimension

        gnn_embeddings = torch.cat(gnn_embeddings, dim=1)  # shape: (batch_size, seq_len, gnn_embedding_size)
        prediction = self.transformer(gnn_embeddings)
        
        return prediction


# In[99]:


### nodal data ###
nodal_inputs_train = torch.stack([ torch.tensor(train_data_np_scaled[i][:,0:64].reshape(6,16,4)) for i in range(len(train_data_np_scaled))])
nodal_inputs_test = torch.stack([ torch.tensor(test_data_np_scaled[i][:,0:64].reshape(6,16,4)) for i in range(len(test_data_np_scaled))])

### edge connections ###
edge_index = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])

### global scalar inputs ### 
global_inputs_train = torch.stack([ torch.tensor(train_data_np_scaled[i][:,64:68]) for i in range(len(train_data_np_scaled))])
global_inputs_test = torch.stack([ torch.tensor(test_data_np_scaled[i][:,64:68]) for i in range(len(test_data_np_scaled))])

### Drag sequence addition ###
drag_sequence_addition_train = torch.stack( [ torch.tensor(train_data_np_scaled[i][:-1,-1]).unsqueeze(-1) for i in range(len(global_inputs_train))] )
drag_sequence_addition_test = torch.stack( [ torch.tensor(test_data_np_scaled[i][:-1,-1]).unsqueeze(-1) for i in range(len(global_inputs_test))] )

### Targets ###
outputs_train = torch.stack([ torch.tensor(train_output[i])[None,None] for i in range(len(train_output))])
outputs_test = torch.stack([ torch.tensor(test_output[i])[None,None] for i in range(len(test_output))])

### Combine output with the history sequence ###
outputs_train = torch.cat((drag_sequence_addition_train,outputs_train), dim=1)
outputs_test = torch.cat((drag_sequence_addition_test,outputs_test), dim=1)


# In[100]:


### Basic Dataloader ###
from torch.utils.data import Dataset, DataLoader 

NUM_GRAPHS_PER_BATCH=16
N_train = -1
N_test = -1

train_loader = DataLoader(list(zip(nodal_inputs_train,global_inputs_train,outputs_train )), 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

test_loader = DataLoader(list(zip(nodal_inputs_test,global_inputs_test,outputs_test)), 
                    batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


# ## Run Training

# In[101]:


model = GNN_Transformer(num_gcn_layers=2, gcn_embedding_size=128, 
                        num_nodes=16, num_features=4, dropout_prob=0.2, heads=2,  
                        input_dim=260, d_model=124, nhead=4, num_layers=2, dim_feedforward=64, max_len=6).float().cuda()

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))


from torch_geometric.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Root mean squared error
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)  

# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

### lr scheduler ###
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=5, verbose=True)

epoch_loss_train = list()
epoch_loss_val = list()
lr_list = list()

import torch
import numpy as np
import os

# Assuming `early_stopping_patience` is defined (number of epochs to wait before stopping)
early_stopping_patience = 15
best_val_loss = float('inf')
patience_counter = 0

# Directory for saving models
os.makedirs(save_loc, exist_ok=True)

for epoch in range(0, 200):
    print(f'Starting Epoch {epoch+1}')

    loss_train = list()
    loss_val = list()
    
    # Training loop
    model.train()
    for nodal, scalar, targets in train_loader:
        
        nodal = nodal.to(torch.float32).cuda()
        scalar = scalar.to(torch.float32).cuda()
        targets = targets.to(torch.float32).cuda()

        optimizer.zero_grad()

        pred = model(nodal, scalar)
        loss = loss_fn(pred, targets.squeeze(-1))
        loss.backward()  
        optimizer.step()

        loss_train.append(loss.item())
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for nodal, scalar, targets in test_loader:
            
            nodal = nodal.to(torch.float32).cuda()
            scalar = scalar.to(torch.float32).cuda()
            targets = targets.to(torch.float32).cuda()
            
            pred = model(nodal, scalar)
            loss = loss_fn(pred, targets.squeeze(-1))
            loss_val.append(loss.item())
    
    train_loss_mean = np.mean(loss_train)
    val_loss_mean = np.mean(loss_val)

    print(f'Epoch {epoch+1} finished with training loss = {train_loss_mean}')
    print(f'Testing loss = {val_loss_mean}\n')

    epoch_loss_train.append(train_loss_mean)
    epoch_loss_val.append(val_loss_mean)

    # Applying learning rate scheduling
    lr_list.append(optimizer.param_groups[0]['lr'])
    scheduler.step(epoch_loss_train[-1])

    # Check for improvement in validation loss
    if val_loss_mean < best_val_loss:
        best_val_loss = val_loss_mean
        torch.save(model.state_dict(), os.path.join(save_loc, "best_model.pt"))
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

    # Save model periodically
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_loc, f"model_{epoch}.pt"))

    np.save(os.path.join(save_loc, "epoch_loss_train.npy"), epoch_loss_train)
    np.save(os.path.join(save_loc, "epoch_loss_val.npy"), epoch_loss_val)

print("Training has completed")


# In[105]:


### Load the model ###
# model.load_state_dict(torch.load("/home/neilashwinraj/gnns/temporal_gnns/simple_connections_data/temporal_split/rho2_20percent_Re50/history_5/results_history_5_seq2seq/model_30"))
# model = model.cuda()


# In[107]:


# train = np.load("/home/neilashwinraj/gnns/temporal_gnns/simple_connections_data/temporal_split/rho2_20percent_Re50/history_5/results_history_5_seq2seq/epoch_loss_train_1.npy")
# test = np.load("/home/neilashwinraj/gnns/temporal_gnns/simple_connections_data/temporal_split/rho2_20percent_Re50/history_5/results_history_5_seq2seq/epoch_loss_test_1.npy")

# plt.semilogy(train)
# plt.semilogy(test)


# In[108]:


# test_loader = DataLoader(list(zip(nodal_inputs_test,global_inputs_test,outputs_test)), 
#                     batch_size=1, shuffle=False)

# k=0

# pred = list()
# gt = list()

# model.eval()
# for nodal,scalar,targets in test_loader:
    
#     print("Data point number",str(k+1))
#     nodal = nodal.to(torch.float32).cuda()
#     scalar = scalar.to(torch.float32).cuda()
#     targets = targets.to(torch.float32).cuda()
        
#     pred.append(model(nodal,scalar).detach())
#     gt.append(targets.detach())
    
#     k=k+1
#     clear_output(wait=True)
    
# ### Extracting the drag force on the last timestep ### 
# gt_final = torch.stack([ gt[i][0,-1].squeeze(-1) for i in range(len(gt))])
# pred_final = torch.stack([ pred[i][0,-1] for i in range(len(pred))])


# In[109]:


# plt.figure(figsize=(10,5))

# plt.scatter(gt_final.detach().cpu().numpy(),
#            torch.stack(pred).detach().cpu().numpy()[:,0,0],alpha=0.2,
#             c=np.arange(len(gt)))

# print(r2_score( gt_final.detach().cpu().numpy(), pred_final.detach().cpu().numpy() ) )

# plt.plot(np.linspace(0,1,10),np.linspace(0,1,10))

