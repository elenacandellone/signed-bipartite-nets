import pickle
import os
import bz2
import random
from scipy import sparse
import numpy as np
import multiprocessing
from tqdm import tqdm
import igraph as ig
from signet.cluster import Cluster 
from graph_tool.all import *

#from signet import SignedNetwork
from src.signet import *

# Ignore FutureWarning
import warnings
warnings.simplefilter('ignore', FutureWarning)


def create_directory(path):

    """Create directory if it doesn't exist."""

    os.makedirs(path, exist_ok=True)


def get_folder(dataset, synth, scenario, method,weighted = True, year = 0):

    """Generate folder path based on input parameters."""

    folder = f'./results/{dataset}/synth/{scenario}/{method}/' if synth else f'./results/{dataset}/real-data/{method}/'
    if year != 0:
        folder = f'./results/{dataset}/synth/{year}/{scenario}/{method}/' if synth else f'./results/{dataset}/real-data/{year}/{method}/'
    folder = folder + f'weighted/' if weighted else folder + f'unweighted/'
    create_directory(folder)
    return folder

def save_pickle(path, filename, data):

    """Save data to a pickle file."""

    if not os.path.exists(path + filename):
        with open(path + filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def spinglass(mat, _gamma_, _lambda_, _run_, weighted, path):

    """Apply Spinglass community detection algorithm."""

    filename = f'gamma_{_gamma_}_lambda_{_lambda_}_run_{_run_}.pickle'
    print(path+filename)
    if not os.path.exists(path+filename): 
        g = ig.Graph.Weighted_Adjacency(mat, mode='undirected') if weighted else ig.Graph.Weighted_Adjacency(sparse.csr_matrix(mat).sign().todense(), mode='undirected')
        g = g.components().giant()
        comm = g.community_spinglass(weights="weight", spins=50, parupdate=False, start_temp=1, 
                                            stop_temp=0.01, cool_fact=0.99, update_rule='simple', gamma= _gamma_, 
                                            implementation='neg', lambda_= _lambda_)
        save_pickle(path, filename, comm.membership)

def sponge(mat, _run_, n_clusters, path):

    """Apply SPONGE community detection algorithm."""

    filename = f'{n_clusters}_run_{_run_}.pickle'
    print(path+filename)
    if not os.path.exists(path+filename):
        A = sparse.csr_matrix(mat).sign()
        c = Cluster((A.multiply(A>0), -A.multiply(A<0)))
        save_pickle(path, filename, c.SPONGE_sym(k=n_clusters, eigens=n_clusters, mi=1e20))

def sbm(mat, path, weighted=True, degree_corrected=False):

    """Apply Stochastic Block Model community detection algorithm."""

    if not degree_corrected:
        filename = f'sbm_not_deg_corr.pkl'
    else:
        filename = f'sbm_deg_corr.pkl'
    print(path+filename)
    if not os.path.exists(path + filename):
        graph_build = SignedNetwork()
        g = graph_build.graph_construction(repre=mat, repre_type='adj', is_directed=check_if_directed(mat))
        if weighted:
            state = minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.absweight_ln, g.ep.sign],
                                                            rec_types=["real-normal", "discrete-binomial"],
                                                            deg_corr=degree_corrected))
        else:
            state = minimize_blockmodel_dl(g, state_args=dict(recs=[g.ep.sign],
                                                            rec_types=["discrete-binomial"],
                                                            deg_corr=degree_corrected))
        for i in tqdm(range(100)):
            ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)
        membership_SBM = list(contiguous_map(state.get_blocks()))
        save_pickle(path, filename, membership_SBM)
