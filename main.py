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
    
    movieMapping = Movie(df_movie)
    userMapping = User(df_rating)
    
    # evaluate normilization
    df_rating['evaluate'].map({'normal': 1, 'good': 2,'bad': 0})
    
    #print(movieMapping.get_movie_name_by_id(11))
    #print(userMapping.get_user_name_by_id(11))
    
    for index , row in df_rating.iterrows():
        df_rating.at[index, 'userID'] = userMapping.get_user_id(row['author'])
        df_rating.at[index, 'movieID'] = movieMapping.get_movie_id(row['movie'])
        
    edge_index = df_rating[["userID", "movieID"]].values.transpose()

    movie_node_features = movieMapping.get_movie_node_features()
    user_node_features = userMapping.get_user_node_features()
    
    #node_features = {
    #    'user': torch.tensor(user_node_features.values),
    #    'movie': torch.tensor(movie_node_features.values)
    #}
    
    #data = HeteroData(node_features=node_features)
    data = HeteroData()
    data['user'].x = torch.tensor(user_node_features.values).float().to(device)
    data['user'].num_nodes = user_node_features.shape[0]
    data['movie'].x = torch.tensor(movie_node_features.values).float().to(device)
    data['movie'].num_nodes = movie_node_features.shape[0]
    
    data['user', 'rating', 'movie'].edge_index = torch.tensor(edge_index).type(torch.int64).to(device)
    data['user', 'rating', 'movie'].edge_label = torch.tensor(df_rating.evaluate.values).to(device)
   

    #weights = df_rating.evaluate.values
    #edge = df_rating[["author", "movie"]].values    
    #draw(userMapping.get_user_name_by_id(''),movieMapping.get_movie_name_by_id(''),edge,weights)

    return data

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
    def __init__(self, hidden_channels):
        super().__init__()
        
        data = readData()
        data = T.ToUndirected()(data)
        
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
def test(model,data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    loss_funxtion = CrossEntropyLoss()
    loss = loss_funxtion(pred, target.long())
    
    accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
    pred_max = torch.argmax(pred, dim=1)
    acc = accuracy(pred_max.to(int), target.to(int))
    
    return loss,acc

def main():
    
    data = readData()
    
    data = T.ToUndirected()(data)
    transform = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        neg_sampling_ratio=0.0,
        edge_types=[('user', 'rating', 'movie')],
        rev_edge_types = [('movie', 'rating', 'user')]
    )
    
    train_data, val_data, test_data = transform(data)
    
    model = Model(hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    
    for epoch in range(1, 41):
        loss,acc = train(model,optimizer,train_data)
        train_loss , train_acc = test(model,train_data)
        val_loss , val_acc = test(model,val_data)
        test_loss,test_acc = test(model,test_data)
        print(f'Epoch: {epoch:03d}, Accuracy: {acc * 100 : .2f}%, Train: {train_acc * 100 : .2f}%, '
              f'Val: {val_acc * 100 : .2f}%, Test: {test_acc * 100 : .2f}%')
        
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_loss:.4f}, '
        #      f'Val: {val_loss:.4f}, Test: {test_loss:.4f}')

if __name__ == "__main__":
    main()
    
    
