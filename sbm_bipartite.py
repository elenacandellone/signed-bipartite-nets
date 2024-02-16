from graph_tool.all import *
from collections import Counter,defaultdict
from tqdm import tqdm
import numpy as np


# ADAPTATION FROM https://github.com/martingerlach/hSBM_Topicmodel
# Import necessary libraries
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.preprocessing import LabelEncoder
import bz2
import pickle

# Define the path to the data directory
path = './data/meneame/real-data/'

# Read votes data from a compressed file
df = pd.read_csv(path + 'df_stories_votes.tsv.gz', sep="\t", compression="gzip").drop_duplicates().reset_index(drop=True)


# Concatenate 'story_id' and 'username_vote' into a single series
all_ids = pd.concat([df['story_id'].astype(str), df['username_vote']])
# Apply LabelEncoder to the concatenated series
encoder = LabelEncoder().fit(all_ids)
# Transform 'story_id' and 'username_vote' using the fitted encoder
df['story_index'] = encoder.transform(df['story_id'].astype(str))
df['username_index'] = encoder.transform(df['username_vote'])

df
edge_list = list(df[['username_index','story_index', 'story_vote']].itertuples(index=False, name=None))
g = Graph(directed=False)
g.add_edge_list(edge_list,  eprops=[('weight', 'int')])
is_bip, part = is_bipartite(g, partition=True)
signs = [0 if w <0 else 1 for w in g.ep.weight.a]
kinds = [1 if part[v] == 1 else 0 for v in g.vertices()]
g.vp['kind'] = g.new_vertex_property("int", vals=kinds)
g.ep['sign'] = g.new_edge_property("int", vals=signs)
clabel = g.vp['kind']
sign_prop = g.ep['sign']

state_args_ = {'clabel': clabel, 'pclabel': clabel}
state = minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.sign], rec_types=["discrete-binomial"],
                                                            deg_corr=True, **state_args_), multilevel_mcmc_args=dict(verbose=True))