#---------------------------------------------- Libraries ----------------------------------------------
from os import path
import pandas as pd
import numpy as np
import os
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import scipy
import scipy.stats as stats
from sklearn.decomposition import PCA

#---------------------------------------------- Class to clean the datasets fetched ----------------------------------------------
# The class receives the raw data from the datasets and performs some preprocessing functions
class Preprocessing():
    def __init__(self) -> None:
        pass
    
    # Function to clean the datasets received as parameters and returns cleaned ones together with a list of empty columns
    # df_r, df_nr: datasets to clean
    def cleaning(self, df_r: pd.DataFrame, df_nr: pd.DataFrame): #-> Tuple[pd.DataFrame, pd.DataFrame, list]:
        # Analysis of empty columns for both dataframes
        list_r = []
        for column in df_r.columns:
            if df_r[column].any() == False:
                list_r.append(column)

        list_nr = []
        for column in df_nr.columns:
            if df_nr[column].any() == False:
                list_nr.append(column)

        common_lists = []
        for i in list_r:
            if i in list_nr:
                common_lists.append(i)

        # Drop the common empty columns
        df_r.drop(columns = common_lists, inplace = True)
        df_nr.drop(columns = common_lists, inplace = True)

        # Delete weird data from the EDAD and TIEMPO_EMP columns
        df_r = df_r.where((df_r['EDAD'] >= 16) & (df_r['TIEMPO_EMP'] >= 0))
        df_nr = df_nr.where((df_nr['EDAD'] >= 16) & (df_nr['TIEMPO_EMP'] >= 0))

        # Drop all the NaN's into both datasets
        df_r.dropna(inplace = True)
        df_nr.dropna(inplace = True)

        return df_r, df_nr, common_lists

    # Function to perform the preparation of the data to having it ready to model training
    # df_r, df_nr: datasets to prepare
    def preparing(self, df_r: pd.DataFrame, df_nr: pd.DataFrame) -> pd.DataFrame:
        # Labelling each dataset
        df_r['LABEL'] = 1
        df_nr['LABEL'] = 0

        # Merging the data
        df_merged = pd.concat([df_r, df_nr], join = 'inner', ignore_index = True)

        # Shuffling and dropping the index
        #df_merged = df_merged.sample(frac = 1)
        df_merged.drop(['PERSONA'], axis = 1, inplace = True)

        return df_merged

    # Function to standarise the data from the input dataset transforming to mu = 0 and std = 1
    # df_m: dataset to standarise
    def standarise(self, df_m: pd.DataFrame):
        labels = df_m['LABEL']
        df_m_scaled = df_m.loc[:, df_m.columns != 'LABEL']

        cols_ = df_m_scaled.columns

        scaler = StandardScaler()
        scaler.fit(df_m_scaled)
        df_m_scaled = scaler.transform(df_m_scaled)

        df_m_scaled = pd.DataFrame(df_m_scaled, columns = cols_)
        df_m_scaled['LABEL'] = labels

        return df_m_scaled, scaler

    # Function to perform the Principal Component Analysis to the input dataset
    # df_ms: dataset to transform through PCA
    # components: number of components to which the df_ms will be transformed
    def pca(self, df_ms: pd.DataFrame, components: int):
        labels = df_ms['LABEL']
        df_pca = df_ms.loc[:, df_ms.columns != 'LABEL']

        pca_ = PCA(n_components = components)
        df_pca = pca_.transform(df_pca)

        df_pca = pd.DataFrame(df_pca)
        df_pca['LABEL'] = labels

        return df_pca, pca_
    
#---------------------------------------------- Calls ----------------------------------------------
prep = Preprocessing()




        
    