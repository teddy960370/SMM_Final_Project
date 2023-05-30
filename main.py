# -*- coding: utf-8 -*-
"""
Created on Sat May 20 23:08:21 2023

@author: ted
"""
import torch
from torch_geometric.nn import SAGEConv, to_hetero,Sequential,Linear
from torch.nn import ReLU,MSELoss,CrossEntropyLoss
from torchmetrics.classification import Accuracy
from torch_geometric.loader import NeighborLoader,HGTLoader

import pandas as pd
from movie import Movie
from user import User
import networkx as nx

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

import torch_geometric.transforms as T
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw(user,movie,edge,weights):
    
    # Create networkx graph
    graph = nx.MultiDiGraph()
    
    # Add nodes to the graph for each node type
    graph.add_nodes_from(user,bipartite=0)
    graph.add_nodes_from(movie,bipartite=1)
    graph.add_edges_from(edge)


    # Set the layout for visualization
    #pos = nx.spring_layout(graph)
    
    # Visualize the graph
    plt.figure(figsize=(64, 48))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 中文字形
    
    nx.set_edge_attributes(graph, values = weights, name = 'weight')
    
    color = list()
    
    for i, n in enumerate(weights):
        if n == 0:
            color.append('green')
        elif n == 1:
            color.append('black')
        else :
            color.append('red')

    graph.edges(data = True)
    
    nx.draw(graph, 
            pos=nx.drawing.layout.bipartite_layout(graph, user), 
            with_labels=True, 
            node_color='lightblue', 
            node_size=200, 
            edge_color=color,
            #edge_color='gray', 
            width=0.5, 
            alpha=0.8)
    
    plt.title('Heterogeneous Graph Visualization')
    plt.show()
    
def readData():
    
    path = './data/'
    
    df_movie = pd.read_csv(path + "movie.csv")
    #df_user = pd.read_csv(path + "user.csv")
    df_rating = pd.read_csv(path + "rating.csv")
    df_pred = pd.read_csv(path + "pred.csv")
    
    # 組合預測資料
    df_rating = pd.concat([df_rating,df_pred])
    
    
    movieMapping = Movie(df_movie)
    userMapping = User(df_rating)
    
    
    
    #print(movieMapping.get_movie_name_by_id(11))
    #print(userMapping.get_user_name_by_id(11))
    
    for index , row in df_rating.iterrows():
        df_rating.at[index, 'userID'] = userMapping.get_user_id(row['author'])
        df_rating.at[index, 'movieID'] = movieMapping.get_movie_id(row['movie'])
        
    # 分開預測資料
    df_pred_test = df_rating[df_rating['evaluate'].isnull()]
    df_rating = df_rating[df_rating['evaluate'].notnull()]
    
    
    edge_index = df_rating[["userID", "movieID"]].values.transpose()
    
    # evaluate normilization
    df_rating['evaluate'].map({'normal': 1, 'good': 2,'bad': 0})

    movie_node_features = movieMapping.get_movie_node_features()
    user_node_features = userMapping.get_user_node_features()
    

    
    # 訓練資料集
    data = HeteroData()
    data['user'].x = torch.tensor(user_node_features.values).float().to(device)
    data['user'].num_nodes = user_node_features.shape[0]
    data['movie'].x = torch.tensor(movie_node_features.values).float().to(device)
    data['movie'].num_nodes = movie_node_features.shape[0]
    
    data['user', 'rating', 'movie'].edge_index = torch.tensor(edge_index).type(torch.int64).to(device)
    data['user', 'rating', 'movie'].edge_label = torch.tensor(df_rating.evaluate.values).to(device)
   
    # 預測資料集
    pred_data = HeteroData() 
    
# =============================================================================
#     temp = user_node_features.values
#     pred_data['user'].x = torch.tensor(temp).float().to(device)
#     pred_data['user'].num_nodes = temp.shape[0]
#     temp = movie_node_features.values
#     pred_data['movie'].x = torch.tensor(temp).float().to(device)
#     pred_data['movie'].num_nodes = temp.shape[0]
#     pred_edge_index = df_pred_test[["userID", "movieID"]].values.transpose()
#     pred_data['user', 'rating', 'movie'].edge_index = torch.tensor(pred_edge_index).type(torch.int64).to(device)
#     
#     df_pred_test['evaluate'] = 0
#     data['user', 'rating', 'movie'].edge_label = torch.tensor(df_pred_test.evaluate.values).to(device)
# 
# =============================================================================
    return data, pred_data

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 3)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z


class Model(torch.nn.Module):
    def __init__(self, hidden_channels,data):
        super().__init__()
        
        
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
    
def train(model,optimizer,train_data):
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'movie'].edge_label_index)
    target = train_data['user', 'movie'].edge_label
    
    loss_funxtion = CrossEntropyLoss()
    loss = loss_funxtion(pred,target.long())
    
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    pred_max = torch.argmax(pred, dim=1)
    acc = accuracy(pred_max.to(int), target.to(int))
    
    loss.backward()
    optimizer.step()
    return float(loss) , acc


@torch.no_grad()
def validation(model,data):
    
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)

    target = data['user', 'movie'].edge_label.float()
    loss_funxtion = CrossEntropyLoss()
    loss = loss_funxtion(pred, target.long())
    
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    pred_max = torch.argmax(pred, dim=1)
    acc = accuracy(pred_max.to(int), target.to(int))
    
    return loss,acc

@torch.no_grad()
def test(model,data):
    
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)

    pred_max = torch.argmax(pred, dim=1)

    return pred_max

def main():
    
    isTest = 0
    
    data, pred_data = readData()
    
    data = T.ToUndirected()(data)
    
    model = Model(hidden_channels=32, data = data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    
    transform = T.RandomLinkSplit(
        num_val = 0.1,
        num_test = 0.0,
        neg_sampling_ratio=0.0,
        is_undirected=True,
        edge_types=[('user', 'rating', 'movie')],
        rev_edge_types = [('movie', 'rating', 'user')]
    )
    

    train_data, val_data, _ = transform(data)
    
    if(isTest == 1):
        pred_data = T.ToUndirected()(pred_data)
        
        transform_pred = T.RandomLinkSplit(
            is_undirected=True,
            neg_sampling_ratio=0.0,
            edge_types=[('user', 'rating', 'movie')],
            rev_edge_types = [('movie', 'rating', 'user')]
        )
        
        test_data = transform_pred(pred_data)
    
    
    
    
    for epoch in range(1, 101):
        train_loss , train_acc = train(model,optimizer,train_data)
        
        if(isTest != 1):
            val_loss , val_acc = validation(model,val_data)
            
            #print(f'Epoch: {epoch:03d}, Accuracy: Train: {train_acc * 100 : .2f}%, '
            #      f'Val: {val_acc * 100 : .2f}%')
            
            print(f'Epoch: {epoch:03d}, Loss: Train: {train_loss:.4f}, '
                  f'Val: {val_loss:.4f}')
            
        else :
            print(f'Epoch: {epoch:03d}, Accuracy:, Train: {train_acc * 100 : .2f}%, ')
            
        
        

    if(isTest == 1):
        result = test(model,test_data)
        
        pred_data.edge_index = result
    
    
    

if __name__ == "__main__":
    main()
    
    
