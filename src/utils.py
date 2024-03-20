import pickle
import os
import bz2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import distinctipy
import matplotlib.cm as cm
import matplotlib.colors as mcolors


from scipy import sparse
import numpy as np
import pandas as pd

import igraph as ig

from sklearn.metrics.cluster import normalized_mutual_info_score, rand_score, adjusted_rand_score, v_measure_score

from src.commdet import create_directory

# change the default options of visualization
text_color = "#404040"

custom_params = {"axes.spines.right": False, "axes.spines.top": False, "axes.spines.left": True, "axes.spines.bottom": True,
                "lines.linewidth": 2, "grid.color": "lightgray", "legend.frameon": True,
                 "xtick.labelcolor": text_color, "ytick.labelcolor": text_color, "xtick.color": text_color, "ytick.color": text_color,"text.color": text_color,
                "axes.labelcolor": text_color, "axes.titlecolor":text_color,"figure.dpi": 200,
                "axes.titlelocation":"center","xaxis.labellocation":"center","yaxis.labellocation":"center"}


#palette = distinctipy.get_colors(21, pastel_factor=0.8)
wong = ['#332288', '#117733','#44AA99', '#88CCEE','#DDCC77', '#CC6677','#AA4499','#882255'  ]

palette = ['#264653', '#2a9d8f', '#f4a261','#e76f51']
sns.set_theme(context='paper', style='white', palette=palette, font_scale=1.1, color_codes=True,
rc=custom_params)

colors = distinctipy.get_colors(9, pastel_factor=0.5)
text_size = 30
lw = 10
scenarios = ['u_np_s_np',  'u_np_s_p', 'u_p_s_np','u_p_s_p']

uniform_scenarios = [f'uniform/{scenario}' for scenario in scenarios]
deg_corr_scenarios = [f'deg_corr/{scenario}' for scenario in scenarios]
titles = ['U NP\nS NP',  'U NP\nS  P', 'U  P\nS NP','U  P\nS  P']
runs = 10

dropbox_folder ='/Users/Cande007/Dropbox/Apps/Overleaf/2023_community_detection_on_signed_networks/'



def network(mat, scenario, labels, weighted=True):
    
    if weighted:
        g = ig.Graph.Weighted_Adjacency(mat, mode = 'undirected')
        g.vs["original_id"] = list(range(g.vcount()))

        #g.simplify(multiple=True, loops=True, combine_edges=dict(weight="sum"))
    else:
        g = ig.Graph.Weighted_Adjacency(sparse.csr_matrix(mat).sign().todense(), mode = 'undirected')
        g.vs["original_id"] = list(range(g.vcount()))
    
    g.vs["labels"] = labels
    #g.vs['color'] = ['#42BFDD' if w > 0 else '#F24333' for w in labels]
    g = g.components().giant()

    return g


def get_folder(dataset, synth, scenario, method,weighted = True, year = 0):
    folder = f'./results/{dataset}/synth/{scenario}/{method}/' if synth else f'./results/{dataset}/real-data/{method}/'
    if year >0:
        folder = f'./results/{dataset}/synth/{year}/{scenario}/{method}/' if synth else f'./results/{dataset}/real-data/{year}/{method}/'
    folder = folder + f'weighted/' if weighted else folder + f'unweighted/'
        #create_directory(folder)
    return folder


def rand_score_df(method, dataset, mats, labels, synth, scenarios,titles=titles, metrics = 'rand', year = 0, weight = True):
    if method == 'spinglass':
        gammas = [0.5, 1, 2]
        lambdas = [0.5, 1, 2]
        dfs = []
        for i in range(len(mats)):
            mat = mats[i]
            label = labels[i]
            scenario = scenarios[i]
            title = titles[i]
            if isinstance(year, list):
                y = year[i]
            else:
                y = year
            g = network(mat, scenario, label)
            for weight in [True,False]:
                df = pd.DataFrame(columns = gammas, index = lambdas)
                for gamma in gammas:
                    for lambd in lambdas:
                        folder = get_folder(dataset,synth, scenario,'spinglass',weighted=weight,year=y)
            
                        with open(folder+f'gamma_{gamma}_lambda_{lambd}_run_0.pickle', 'rb') as f:
                            spinglass = pickle.load(f)
                       
                        if metrics == 'rand':
                            df[gamma][lambd] = rand_score(g.vs['labels'], spinglass)
                        elif metrics == 'ari':
                            df[gamma][lambd] = adjusted_rand_score(g.vs['labels'], spinglass)
                        elif metrics == 'nmi':
                            df[gamma][lambd] = normalized_mutual_info_score(g.vs['labels'], spinglass)
                        elif metrics == 'v_score':
                            df[gamma][lambd] = v_measure_score(g.vs['labels'], spinglass)
                dfs.append(df)
        return dfs
    
    elif method == 'sponge':
        ks = [1,2,3,4,5,6,7,8,9,10]
        df = pd.DataFrame(columns = ks, index = titles)
        df_std = pd.DataFrame(columns = ks, index = titles)
        for i in range(len(mats)):
            label = labels[i]
            scenario = scenarios[i]
            title = titles[i]
            if isinstance(year, list):
                y = year[i]
            else:
                y = year
            g = network(mats[i], scenario, label)
            
            folder = get_folder(dataset,synth, scenario,'sponge',weighted=weight,year=y)
            
            for k in ks:
                rand = []
                for run in range(runs):
                    with open(folder+f'{k}_run_{run}.pickle', 'rb') as f:
                        sponge = pickle.load(f)
                    if metrics == 'rand':
                        rand.append(rand_score(label, sponge))
                    elif metrics == 'ari':
                        rand.append(adjusted_rand_score(label, sponge))
                    elif metrics == 'nmi':
                        rand.append(normalized_mutual_info_score(label, sponge))
                    elif metrics == 'v_score':
                        rand.append(v_measure_score(label, sponge))
                            
                df[k][title] = np.mean(rand)
                df_std[k][title] = np.std(rand)
        return df, df_std
    
    elif method == 'sbm':
        ks = [1,2,3,4,5,6,7,8,9,10]
        df = pd.DataFrame(columns = ks, index = titles)
        #df_std = pd.DataFrame(columns = ks, index = titles)
        for i in range(len(mats)):
            label = labels[i]
            scenario = scenarios[i]
            title = titles[i]
            if isinstance(year, list):
                y = year[i]
            else:
                y = year
            g = network(mats[i], scenario, label)
            
            folder = get_folder(dataset,synth, scenario,'sbm',weighted=weight,year=y)
            for k in ks:
                rand = []
                with open(folder+f'sbm_deg_corr_{k}.pkl', 'rb') as f:
                    sponge = pickle.load(f)
                if metrics == 'rand':
                    rand.append(rand_score(label, sponge))
                elif metrics == 'ari':
                    rand.append(adjusted_rand_score(label, sponge))
                elif metrics == 'nmi':
                    rand.append(normalized_mutual_info_score(label, sponge))
                elif metrics == 'v_score':
                    rand.append(v_measure_score(label, sponge))          
                df[k][title] = np.mean(rand)
                #df_std[k][title] = np.std(rand)
        return df
    


def load_data(path):
    path_adj = path+'/adj/'
    path_labels = path+'/label/'
    file_names = ['u_np_s_np.pkl',  'u_np_s_p.pkl', 'u_p_s_np.pkl','u_p_s_p.pkl']
    
    mats = [pickle.load(open(path_adj + f, 'rb')) for f in file_names]
    labels = [pickle.load(open(path_labels + f, 'rb')) for f in file_names]
    return mats, labels

def plot_spinglass_synth(ax, dfs, lambdas, gammas, titles, text_size, xticks = True):
    x = 0
    ticks = []
    for lambd in lambdas:
        for gamma in gammas:
            delta = 0
            for i, df in enumerate(dfs):
                ax.scatter(x+delta, df.at[gamma, lambd], s = 500, marker = 'h')
                delta += 0.15
            ticks.append(f'$\gamma^+$ = {gamma:.1f}\n$\gamma^-$ = {lambd:.1f}')
            ax.axvline(x+0.7, ls='--', alpha =0.2)
            x +=1
    if xticks:
        ax.set_xticks([i+0.2 for i in range(x)],ticks, fontsize=text_size, rotation =90)
    else:
        ax.set_xticks([])
    #ax.tick_params(axis='both', labelsize=text_size)
    #patchList = [mpatches.Patch(color=palette[key], label=title) for key, title in enumerate(titles)]
    #plt.legend(handles=patchList, bbox_to_anchor=(1, 1), fontsize = text_size)

def plot_sponge_synth(ax, df, df_std, text_size, titles,  xticks = True):
    x = df.T.index.tolist()
    mean = df.T.values.tolist()
    std = df_std.T.values.tolist()
    ax.plot(x, [row[0] for row in mean], label = df.T.columns[0],marker='s', markersize=15, linestyle='-', linewidth=lw, alpha =0.8)
    ax.plot(x, [row[1:] for row in mean], label = df.T.columns[1:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8)
    for m, s, in zip(np.array(mean).T, np.array(std).T):
        m = np.array(m)
        s = np.array(s)
        ax.fill_between(x, m-s, m+s, alpha=0.4, interpolate=True)
    ax.scatter(x = 1, y = 1, s = 70, marker = 'x', c = '#219ebc', zorder=3, label = 'exp. value\nusers pol.')
    ax.scatter(x = 2, y = 1, s = 70, marker = 'x', c = '#bc6c25', zorder=3, label = 'exp. value\nusers not pol.')
    
    if xticks:
        ax.set_xlabel('no. of clusters', fontsize =text_size)
        ax.set_xticks(x, x, fontsize=text_size)
    else:
        ax.set_xticks([])
    ax.tick_params(axis='both', labelsize=text_size)
    patchList = [mpatches.Patch(color=palette[key], label=title) for key, title in enumerate(titles)]
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), fontsize = text_size)


def plot_sbm_synth(ax, df, text_size, titles,  xticks = True):
    x = df.T.index.tolist()
    mean = df.T.values.tolist()
    #std = df_std.T.values.tolist()
    ax.plot(x, [row[0] for row in mean], label = df.T.columns[0],marker='s', markersize=15, linestyle='-', linewidth=lw, alpha =0.8)
    ax.plot(x, [row[1:] for row in mean], label = df.T.columns[1:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8)
    #ax.scatter(x = 1, y = 1, s = 70, marker = 'x', c = '#219ebc', zorder=3, label = 'exp. value\nusers pol.')
    #ax.scatter(x = 2, y = 1, s = 70, marker = 'x', c = '#bc6c25', zorder=3, label = 'exp. value\nusers not pol.')
    
    if xticks:
        ax.set_xlabel('no. of clusters', fontsize =text_size)
        ax.set_xticks(x, x, fontsize=text_size)
    else:
        ax.set_xticks([])
    ax.tick_params(axis='both', labelsize=text_size)
    patchList = [mpatches.Patch(color=palette[key], label=title) for key, title in enumerate(titles)]
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), fontsize = text_size)

def plot_spinglass_synth_split(ax1,ax2, dfs, lambdas, gammas, titles, text_size, xticks = True):
    x = 0
    ticks = []
    for lambd in lambdas:
        for gamma in gammas:
            delta = 0
            for i, df in enumerate(dfs):
                if i<2:
                    ax1.scatter(x+delta, df.at[gamma, lambd], s = 500, marker = 'h', c=palette[i])
                    delta += 0.15
                else:
                    ax2.scatter(x+delta, df.at[gamma, lambd], s = 500, marker = 'h', c=palette[i])
                    delta += 0.15
            ticks.append(f'$\gamma^+$ = {gamma:.1f}\n$\gamma^-$ = {lambd:.1f}')
            ax1.axvline(x+0.7, ls='--', alpha =0.2)
            ax2.axvline(x+0.7, ls='--', alpha =0.2)
            x +=1
    if xticks:
        ax1.set_xticks([i+0.2 for i in range(x)],ticks, fontsize=text_size, rotation =90)
        ax2.set_xticks([i+0.2 for i in range(x)],ticks, fontsize=text_size, rotation =90)
    else:
        ax1.set_xticks([])
        ax1.set_xticks([])
    ax2.tick_params(axis='both', labelsize=text_size)
    patchList = [mpatches.Patch(color=palette[key], label=title) for key, title in enumerate(titles)]
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), fontsize = text_size)

def plot_sponge_synth_split(ax1, ax2, df, df_std, text_size, titles,  xticks = True):
    x = df.T.index.tolist()
    mean = df.T.values.tolist()
    std = df_std.T.values.tolist()
    ax1.plot(x, [row[0] for row in mean], label = df.T.columns[0],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[0])
    ax1.plot(x, [row[1] for row in mean], label = df.T.columns[1:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[1])


    ax2.plot(x, [row[2] for row in mean], label = df.T.columns[2:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[2])
    ax2.plot(x, [row[3] for row in mean], label = df.T.columns[3:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[3])
    en=0
    for m, s, in zip(np.array(mean).T, np.array(std).T):
        m = np.array(m)
        s = np.array(s)
        if en<2:
            ax1.fill_between(x, m-s, m+s, alpha=0.4, interpolate=True, color=palette[en])
        else:
            ax2.fill_between(x, m-s, m+s, alpha=0.4, interpolate=True, color=palette[en])
        en+=1
    ax1.scatter(x = 1, y = 1, s = 200, marker = 'x', c = '#219ebc', zorder=1, label = 'exp. value\nusers pol.')
    ax2.scatter(x = 2, y = 1, s = 200, marker = 'x', c = '#bc6c25', zorder=1, label = 'exp. value\nusers not pol.')
    
    if xticks:
        ax1.set_xlabel('no. of clusters', fontsize =text_size)
        ax1.set_xticks(x, x, fontsize=text_size)
        ax2.set_xlabel('no. of clusters', fontsize =text_size)
        ax2.set_xticks(x, x, fontsize=text_size)
    else:
        ax1.set_xticks([])
        ax2.set_xticks([])
    ax1.tick_params(axis='both', labelsize=text_size)
    ax2.tick_params(axis='both', labelsize=text_size)
    patchList = [mpatches.Patch(color=palette[key], label=title) for key, title in enumerate(titles)]
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), fontsize = text_size)

def plot_sbm_synth_split(ax1, ax2, df, text_size, titles,  xticks = True):
    x = df.T.index.tolist()
    mean = df.T.values.tolist()

    ax1.plot(x, [row[0] for row in mean], label = df.T.columns[0],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[0])
    ax1.plot(x, [row[1] for row in mean], label = df.T.columns[1:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[1])


    ax2.plot(x, [row[2] for row in mean], label = df.T.columns[2:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[2])
    ax2.plot(x, [row[3] for row in mean], label = df.T.columns[3:],marker='o', markersize=5, linestyle='-', linewidth=lw, alpha =0.8, c=palette[3])
    
    #ax1.scatter(x = 1, y = 1, s = 200, marker = 'x', c = '#219ebc', zorder=1, label = 'exp. value\nusers pol.')
    #ax2.scatter(x = 2, y = 1, s = 200, marker = 'x', c = '#bc6c25', zorder=1, label = 'exp. value\nusers not pol.')
    
    if xticks:
        ax1.set_xlabel('no. of clusters', fontsize =text_size)
        ax1.set_xticks(x, x, fontsize=text_size)
        ax2.set_xlabel('no. of clusters', fontsize =text_size)
        ax2.set_xticks(x, x, fontsize=text_size)
    else:
        ax1.set_xticks([])
        ax2.set_xticks([])
    ax1.tick_params(axis='both', labelsize=text_size)
    ax2.tick_params(axis='both', labelsize=text_size)
    patchList = [mpatches.Patch(color=palette[key], label=title) for key, title in enumerate(titles)]
    plt.legend(handles=patchList, bbox_to_anchor=(1, 1), fontsize = text_size)


def save_plot(path, filename, dropbox_folder=dropbox_folder):
    create_directory(path)
    create_directory(path.replace('./', f'{dropbox_folder}'))
    plt.savefig(path+filename)
    plt.savefig(path.replace('./', f'{dropbox_folder}')+filename)
