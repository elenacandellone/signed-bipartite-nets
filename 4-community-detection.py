import numpy as np
import os
import random
import pickle
from joblib import Parallel, delayed
from src.commdet import *

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

n_jobs = 4
n_runs = 1
n_runs_sponge = 10
gammas = [0.5, 1, 2]
lambdas = [0.5, 1, 2]
n_clusters = [1,2,3,4,5,6,7,8,9,10]
scenarios = ['u_np_s_np', 'u_p_s_np', 'u_np_s_p', 'u_p_s_p']

def load_matrices(path, scenarios=scenarios):
    return [pickle.load(open(f'{path}{scenario}.pkl', 'rb')) for scenario in scenarios]

def run_spinglass(mats, dataset, synth, scenarios, gammas=gammas, lambdas=lambdas, years = None):

    """Run Spinglass community detection algorithm in parallel."""

    if years is None:
        years_ = [0]*len(mats)
    else:
        years_ = years
    Parallel(n_jobs=n_jobs)(delayed(spinglass)(mat=mats[i], _gamma_=gamma, 
                                          _lambda_=lambd, _run_=run, weighted=weight, 
                                          path = get_folder(dataset,synth, scenarios[i],'spinglass',weight,year=years_[i])) for i in range(len(mats)) for run in range(n_runs) for gamma in gammas for lambd in lambdas for weight in [True, False])

def run_sponge(mats, dataset, synth, scenarios, n_clusters =n_clusters, years = None):

    """Run Sponge community detection algorithm in parallel."""

    if years is None:
        years_ = [0]*len(mats)
    else:
        years_ = years
    Parallel(n_jobs=n_jobs)(delayed(sponge)(mat=mats[i], _run_=run, n_clusters=n, 
                                            path = get_folder(dataset,synth, scenarios[i],'sponge',year = years_[i])) for i in range(len(mats)) for run in range(n_runs_sponge) for n in n_clusters)

def run_sbm(mats, dataset, synth, scenarios, years = None):

    """Run Stochastic Block Model community detection algorithm in parallel."""

    if years is None:
        years_ = [0]*len(mats)
    else:
        years_ = years
    Parallel(n_jobs=n_jobs)(delayed(sbm)(mat = mats[i], weighted = weight, 
                                         degree_corrected = deg_correct, path = get_folder(dataset,synth, scenarios[i],'sbm',weighted=weight,year=years_[i])) for i in range(len(mats)) for weight in [True,False] for deg_correct in [True,False])

# COMMUNITY DETECTION - MENEAME - SYNTHETIC NETWORKS
mats = load_matrices('./data/meneame/synth/uniform/adj/', scenarios=scenarios)
uniform_scenarios = [f'uniform/{scenario}' for scenario in scenarios]

run_spinglass(mats, dataset='meneame', synth=True, scenarios=uniform_scenarios)
run_sponge(mats, dataset='meneame', synth=True, scenarios=uniform_scenarios)
run_sbm(mats, dataset='meneame', synth=True, scenarios=uniform_scenarios)

# DEGREE CORRECTED VERSION
mats = load_matrices('./data/meneame/synth/deg_corr/adj/', scenarios=scenarios)
deg_corr_scenarios = [f'deg_corr/{scenario}' for scenario in scenarios]

run_spinglass(mats, dataset='meneame', synth=True, scenarios=deg_corr_scenarios)
run_sponge(mats, dataset='meneame', synth=True, scenarios=deg_corr_scenarios)
run_sbm(mats, dataset='meneame', synth=True, scenarios=deg_corr_scenarios)


# COMMUNITY DETECTION - MENEAME - REAL DATA
with bz2.BZ2File('./data/meneame/real-data/adj_data.pkl', 'r') as f:
    A = pickle.load(f)

A = A.todense()
np.fill_diagonal(A, 0)

run_spinglass([A], dataset='meneame', synth=False, scenarios=['real-data'],gammas=[0.5, 1], lambdas=[0.5, 1])
run_sponge([A],dataset='meneame', synth=False, scenarios=['real-data'] )
run_sbm([A], dataset='meneame', synth=False, scenarios=['real-data'] )



# COMMUNITY DETECTION - HOUSE OF REPRESENTATIVES - SYNTHETIC NETWORKS
for y in np.arange(1990,2023):
    path_adj = f'./data/us-house/synth/uniform/{y}/adj/'
    mats = load_matrices(path_adj)
    
    run_spinglass(mats, dataset='us-house', synth=True, scenarios=scenarios, years=[f'{y}']*len(scenarios))
    run_sponge(mats, dataset='us-house', synth=True, scenarios=scenarios, years=[f'{y}']*len(scenarios))
    run_sbm(mats, dataset='us-house', synth=True, scenarios=scenarios,  years=[f'{y}']*len(scenarios))

# COMMUNITY DETECTION - HOUSE OF REPRESENTATIVES - REAL DATA
path_adj = f'./data/us-house/real-data/adj/'
mats = [pickle.load(open(path_adj + f'{year}.pkl', 'rb')) for year in np.arange(1990,2023)]
scenarios = np.arange(1990,2023)

run_spinglass(mats, dataset='us-house', synth=False, scenarios=scenarios, years=scenarios)
run_sponge(mats, dataset='us-house', synth=False, scenarios=scenarios, years=scenarios)
run_sbm(mats, dataset='us-house', synth=False, scenarios=scenarios, years=scenarios)