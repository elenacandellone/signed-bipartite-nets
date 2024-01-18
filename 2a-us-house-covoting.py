import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import igraph as ig

from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

import os
import pickle
import warnings
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
    csr_matrix, isspmatrix, dok_matrix, lil_matrix, bsr_matrix)
warnings.simplefilter('ignore',SparseEfficiencyWarning)

path = './data/us-house/real-data/'

def create_directory(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def incidence_matrix(df):
    incidence_edgelist = df[['person_encoded', 'call_encoded', 'weight']]
    incidence_matrix_df = incidence_edgelist.pivot(index='person_encoded', columns='call_encoded', values='weight').fillna(0)
    #print(incidence_matrix_df)
    incidence_matrix = csr_matrix(incidence_matrix_df.values)
    return incidence_matrix, incidence_matrix_df.index

df = pd.read_csv(path+'clerk_data.tsv.gz', sep='\t', header= None,names=["person", "person_", "party", "state", "state_short", "vote","year","bill"], compression='gzip')
df = df.dropna()
df = df.drop(columns=['person_'])
df['vote'] = df['vote'].replace(['Aye', 'Yea'], 'yes')
df['vote'] = df['vote'].replace(['Nay', 'No'], 'no')
df['vote'] = df['vote'].replace(['Not Voting'], 'not_voting')
df['vote'] = df['vote'].replace(['Present'], 'present')
df['call_year'] = df['year'].astype(str) + '_' + df['bill'].astype(str)
df['person_party'] = df['person'].astype(str) + '_' + df['party'].astype(str)
df['call_encoded'] = LabelEncoder().fit_transform(df['call_year'])
df['person_encoded'] = LabelEncoder().fit_transform(df['person_party'])
df['party_encoded'] = LabelEncoder().fit_transform(df['party'])

df['weight'] = df['vote'].replace(['yes', 'no', 'not_voting', 'present'], [1, -1, 0, 0])
df = df.loc[df['year'] != 2023]

political_affiliation = dict(zip(df.person_encoded,df.party))

unique_years = np.arange(1990,2023)

df_stats = pd.DataFrame(columns=['year', 'n_nodes', 'n_edges'])

for year in unique_years:
    df_year = df[df['year'] == year]

    incidence_matrix_year, congress_members = incidence_matrix(df_year)

    A = incidence_matrix_year @ (incidence_matrix_year.T)
    
    A.setdiag(0)
    A = A.todense()
    g = ig.Graph.Weighted_Adjacency(A, mode='undirected') 
    df_stats = pd.concat([df_stats, pd.DataFrame([{'year': year, 'n_nodes': g.vcount(), 'n_edges': g.ecount()}])], ignore_index=True)


    adj_path = path + 'adj/'
    create_directory(adj_path)
    labels_path = path + 'label/'
    create_directory(labels_path)
    filename = f'{year}.pkl'
    if not os.path.exists(adj_path + filename):
        with open(adj_path + filename, 'wb') as f:
            pickle.dump(A, f)

    labels = [political_affiliation[label] for label in congress_members]

    if not os.path.exists(labels_path + filename):
        with open(labels_path +filename, 'wb') as f:
            pickle.dump(labels, f)


print(df_stats)