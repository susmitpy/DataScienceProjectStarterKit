#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:11:12 2020

@author: susmitvengurlekar
"""

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from common_transformers import PassThrough

# Template
class ColNameProcessingTemplate(TransformerMixin,BaseEstimator):
    def __init__(self,return_whole_df):
        self.return_whole_df = return_whole_df
        self.cols = []
        self.added_cols = []

    def fit(self,df,y=None):
        self.cols = list(df.columns)
        return self

    def transform(self,df,*_):
        copy = df.copy()


        if self.return_whole_df:
            return copy
        return copy[self.added_cols]

    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols
        return self.added_cols

class FeatureGenerator(TransformerMixin,BaseEstimator):
    def __init__(self,return_whole_df):
        self.return_whole_df = return_whole_df
        self.cols = cols
        self.added_cols = []


    def fit(self,df,y=None):
        self.cols = list(df.columns)
        return self

    def transform(self,df,*_):
        copy = df.copy()

        # Do the stuff

        # Remove original cols

        if self.return_whole_df:
            return copy
        return copy[self.added_cols]
        
    def get_feature_names(self):
        if self.return_whole_df:
            return self.cols + self.added_cols

        return self.added_cols

def get_column_transformer():
    return ColumnTransformer([
        (),
        ("pass",PassThrough(),["pass columns"])
    ])