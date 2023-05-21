# -*- coding: utf-8 -*-
"""
Created on Sat May 20 23:08:21 2023

@author: ted
"""
import pandas as pd
from movie import Movie
from user import User
import networkx as nx

from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

def draw(data):
    
    # Create networkx graph
    graph = nx.MultiDiGraph()
    
    # Add nodes to the graph for each node type
    graph.add_nodes_from(range(data['user'].num_nodes),bipartite=0)
    graph.add_nodes_from(range(data['movie'].num_nodes),bipartite=1)
    graph.add_edges_from(data['user', 'rating', 'movie'].edge_index.transpose())


    # Set the layout for visualization
    pos = nx.spring_layout(graph)
    
    # Visualize the graph
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos=pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', width=0.5, alpha=0.8)
    plt.title('Heterogeneous Graph Visualization')
    plt.show()
    
def readData():
    
    path = './data/'
    
    df_movie = pd.read_csv(path + "movie.csv")
    #df_user = pd.read_csv(path + "user.csv")
    df_rating = pd.read_csv(path + "rating.csv")
    
    movieMapping = Movie(df_movie)
    userMapping = User(df_rating)
    
    #print(movieMapping.get_movie_name_by_id(11))
    #print(userMapping.get_user_name_by_id(11))
    
    for index , row in df_rating.iterrows():
        df_rating.at[index, 'userID'] = userMapping.get_user_id(row['author'])
        df_rating.at[index, 'movieID'] = movieMapping.get_movie_id(row['movie'])
        
    edge_index = df_rating[["userID", "movieID"]].values.transpose()

    movie_node_features = movieMapping.get_movie_node_features()
    user_node_features = userMapping.get_user_node_features()
    
    node_features = {
        'user': user_node_features,
        'movie': movie_node_features
    }
    
    data = HeteroData(node_features=node_features)
    #data['user'].x = user_node_features
    #data['movie'].x = movie_node_features
    data['user', 'rating', 'movie'].edge_index = edge_index
    #data['user', 'movie'].y = df_rating['']

    data['user'].num_nodes = user_node_features.shape[0]
    data['movie'].num_nodes = movie_node_features.shape[0]
    
    draw(data)


def main():
    readData()

if __name__ == "__main__":
    main()