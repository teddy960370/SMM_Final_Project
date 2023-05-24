# -*- coding: utf-8 -*-
"""
Created on Sun May 21 00:09:24 2023

@author: ted
"""

import pandas as pd

class User:
    def __init__(self, dataframe):
        
        dataframe['evaluate'] = dataframe['evaluate'].map({'normal': 0, 'good': 1, 'bad':-1})
        
        temp = dataframe[['author', 'evaluate']].groupby(['author']).agg(['mean', 'count'])
        temp = temp.reset_index(drop=False)
        temp['index'] = temp.index
        
        temp.columns = ['author','mean','count','index']
        self.df = temp
        
    def get_user_id(self, user_name):
        try:
            user_id = self.df.loc[self.df['author'] == user_name, 'index'].values[0]
            return user_id
        except IndexError:
            return "User not found."

    def get_user_number(self):
        return len(self.df.index)

    def get_user_name_by_id(self, user_id):
        try:
            if(user_id == ''):
                return self.df['author']
            else :
                user_name = self.df.loc[self.df['index'] == user_id, 'author'].values[0]
                return user_name
        except IndexError:
            return "User not found."
        
    def get_user_node_features(self):
        
        user_node_features = self.df[['mean','count']]
        return user_node_features