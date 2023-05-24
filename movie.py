# -*- coding: utf-8 -*-
"""
Created on Sat May 20 23:48:42 2023

@author: ted
"""

import pandas as pd

class Movie:
    def __init__(self, dataframe):

        self.df = dataframe

    def get_movie_id(self, movie_name):
        try:
            movie_id = self.df.loc[self.df['MovieName'] == movie_name, 'MovieID'].values[0]
            return movie_id
        except IndexError:
            return "Movie not found."

    def get_movie_name_by_id(self, movie_id):
        try:
            if(movie_id == ''):
                return self.df['MovieName']
            else :
                movie_name = self.df.loc[self.df['MovieID'] == movie_id, 'MovieName'].values[0]
                return movie_name
        except IndexError:
            return "Movie not found."
        
    def get_movie_number(self):
        return len(self.df.index)
        
    def get_movie_node_features(self):
        
        node_features = self.df
        
        # one hot encoding
        MovieCountry = pd.get_dummies(node_features['MovieCountry'])
        Applicant = pd.get_dummies(node_features['Applicant'])
        Produce = pd.get_dummies(node_features['Produce'])
        
        # concat
        node_features = pd.concat([node_features, MovieCountry,Applicant,Produce], axis=1)
        #node_features = pd.concat([node_features, Applicant], axis=1)
        #node_features = pd.concat([node_features, Produce], axis=1)
        
        # drop useless column
        node_features = node_features.drop(['MovieID','MovieName','ReleaseDate','MovieCountry','Applicant','Produce'],axis = 1)
        
        
        return node_features