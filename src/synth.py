# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import numba
import seaborn as sns
import pandas as pd
import tldextract
import json
from scipy.sparse import csr_matrix, coo_matrix
import pickle
import scipy.stats as stats
from scipy import optimize
import random
import matplotlib.patches as mpatches
import os
from tqdm import tqdm

# Define custom visualization parameters
text_color = "#404040"
custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "lines.linewidth": 2,
    "grid.color": "lightgray",
    "legend.frameon": True,
    "xtick.labelcolor": text_color,
    "ytick.labelcolor": text_color,
    "xtick.color": text_color,
    "ytick.color": text_color,
    "text.color": text_color,
    "axes.labelcolor": text_color,
    "axes.titlecolor": text_color,
    "figure.dpi": 150,
    "axes.titlelocation": "center",
    "xaxis.labellocation": "center",
    "yaxis.labellocation": "center"
}

# Define a custom color palette
wong = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499', '#882255']
sns.set_theme(context='paper', style='white', palette=wong, font_scale=1.1, color_codes=True, rc=custom_params)


import scipy.integrate as spi

################################
# SAMPLE FROM GAUSSIAN MIXTURE #
################################

# Import GaussianMixture and create a function to sample data from a Gaussian mixture
from sklearn.mixture import GaussianMixture
def gaussian_mix(A, mu_a, sigma_a, mu_b, sigma_b, n_samples):

    _weights = np.array([A, 1-A])
    data_gmm = GaussianMixture(n_components=2, covariance_type='spherical')
    data_gmm.weights_ = _weights / sum(_weights) #norm
    data_gmm.means_ = np.array([[mu_a],[mu_b]])
    data_gmm.covariances_ = np.array([sigma_a,  sigma_b])  #keeping the same std for both

    data=data_gmm.sample(n_samples)[0].reshape(1, -1)[0]

    if mu_a == mu_b:
        labels = np.zeros(n_samples)
    else:
        labels=data_gmm.sample(n_samples)[1]

    return data, labels

# Define a function to calculate the difference between two bimodal Gaussian mixtures
def difference_gaussian_mixtures(x, A, B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4, cdf=True):
                                 
    norm = (A * B) + (A * (1-B)) + ((1-A) * B) + ((1-A) * (1-B)) 

    if cdf:
        return (1/norm) * (((A * B) * stats.norm.cdf(x, mu_1 - mu_3, np.sqrt(sigma_1**2 + sigma_3**2)))+
                       ((A * (1-B)) * stats.norm.cdf(x, mu_1 - mu_4, np.sqrt(sigma_1**2 + sigma_4**2))) +
                       (((1-A) * B) * stats.norm.cdf(x, mu_2 - mu_3, np.sqrt(sigma_2**2 + sigma_3**2))) +
                       (((1-A) * (1-B)) * stats.norm.cdf(x, mu_2 - mu_4, np.sqrt(sigma_2**2 + sigma_4**2)))
                        ) 
    else:
        return (1/norm) * (((A * B) * stats.norm.pdf(x, mu_1 - mu_3, np.sqrt(sigma_1**2 + sigma_3**2)))+
                       ((A * (1-B)) * stats.norm.pdf(x, mu_1 - mu_4, np.sqrt(sigma_1**2 + sigma_4**2))) +
                       (((1-A) * B) * stats.norm.pdf(x, mu_2 - mu_3, np.sqrt(sigma_2**2 + sigma_3**2))) +
                       (((1-A) * (1-B)) * stats.norm.pdf(x, mu_2 - mu_4, np.sqrt(sigma_2**2 + sigma_4**2)))
                        )


###########################
# THRESHOLD DETERMINATION #
###########################

# Define a function to calculate the probabilities of positive and negative voting
def prob_voting(df_votes, queue=False, meneame=True):
    with open('./data/meneame/real-data/queue.json', 'r') as f:
        queue = json.load(f)
    if meneame == True:
        df_votes['queue'] = df_votes['story_id'].apply(lambda x: queue.get(str(x)))

        if queue == True:
            df_votes = df_votes.loc[df_votes['queue'] == True]
        else:
            df_votes = df_votes.loc[df_votes['queue'] == False]
    
    n_stories = len(df_votes.story_id.unique().tolist())
    v_plus = df_votes.loc[df_votes['story_vote'] > 0].groupby(by='username_vote').count()['story_vote']
    v_minus = df_votes.loc[df_votes['story_vote'] < 0].groupby(by='username_vote').count()['story_vote']

    return v_plus.mean()/ n_stories, v_minus.mean()/ n_stories


# Define functions to find positive and negative thresholds for voting
def find_thr(A, B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4, v, x0, plus):
    if plus:
        v_plus = v
        # F(t_+) - F(-t_+) = v_+
        def integrand_plus(x, A, B, 
                    mu_1, sigma_1, 
                    mu_2, sigma_2, 
                    mu_3, sigma_3, 
                    mu_4, sigma_4, v_plus):

            return difference_gaussian_mixtures(x, A, B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4)-difference_gaussian_mixtures(-x, A, B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4)-v_plus
        threshold = 0
        threshold = optimize.fsolve(integrand_plus, x0, args=(A, B, 
                                                                mu_1, sigma_1, 
                                                                mu_2, sigma_2, 
                                                                mu_3, sigma_3, 
                                                                mu_4, sigma_4, v_plus))[0]
    else:
        v_minus = v
        # F(-t_-) = v_-/2
        def integrand_minus(x, A, B, 
                    mu_1, sigma_1, 
                    mu_2, sigma_2, 
                    mu_3, sigma_3, 
                    mu_4, sigma_4, v_minus):

            return difference_gaussian_mixtures(x, A, B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4)-(v_minus/2)
        threshold = 0
        threshold = optimize.fsolve(integrand_minus, x0, args=(A, B, 
                                                            mu_1, sigma_1, 
                                                            mu_2, sigma_2, 
                                                            mu_3, sigma_3, 
                                                            mu_4, sigma_4,v_minus ))[0]
        threshold = -threshold

    if (threshold == 0):
        print('threshold not found!')
    return threshold


def find_threshold(A, B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4, v_plus, v_minus, x0_plus, x0_minus):
    threshold_plus = find_thr( A,  B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4, v_plus, x0_plus, plus = True)
    threshold_minus = find_thr( A,  B, mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, mu_4, sigma_4, v_minus, x0_minus, plus = False)
    
    return threshold_plus, threshold_minus


# Define a function to determine thresholds for different scenarios
def thresholds_scenarios(amplitude, sigma, v_pos, v_neg):
    t_p = dict()
    t_m = dict()
    scenarios = [[0,0], [0,1], [1,0], [1,1]]
    x0s = [[0,-0.5], [1,-2] , [1, -2], [0, -2.5]]
    for i, scenario in enumerate(scenarios):
        dist_users = scenario[0]
        dist_stories = scenario[1]
        x0 = x0s[i] 
        t_plus, t_minus = find_threshold(A = amplitude, B = amplitude, mu_1 = - dist_stories, mu_2 = + dist_stories, mu_3 = - dist_users, mu_4 = + dist_users, sigma_1 = sigma, sigma_2 = sigma, sigma_3 = sigma, sigma_4 = sigma, v_plus = v_pos, v_minus = v_neg, x0_plus = abs(dist_stories-dist_users), x0_minus = -abs(dist_stories+dist_users) )
        print(f't_+ = {t_plus}')
        print(f't_- = {t_minus}\n')
        t_p[tuple(scenario)] = t_plus
        t_m[tuple(scenario)] = t_minus

    return t_p, t_m

##########################    
# SAMPLE DEGREE SEQUENCE #
##########################
def sample_deg_sequence(deg_p, deg_m , num_users):

    idx = np.random.choice(np.arange(0,num_users), num_users, replace=False)
    sample_p = deg_p[idx]
    sample_m = deg_m[idx]

    return sample_p, sample_m

################################
# USER-STORY BIPARTITE NETWORK #
################################

# Define a function to determine how users vote on stories
@numba.jit(nopython=True, parallel=False)
def vote(u, s, t_plus, t_minus):
    if np.abs(u-s) < t_plus:
        return 1
    elif np.abs(u-s) > t_minus:
        return -1
    else:
        return 0



# Define a function to fill an incidence matrix based on user votes and story characteristics
@numba.jit(nopython=True, parallel=False)
def fill_mat(users, stories, t_p, t_m, deg_seq_pos, deg_seq_neg, deg_corr = False):
    mat = np.zeros((len(users), len(stories)))
    for user_index, user_ideology in enumerate(users):
        if deg_corr == True:
            pos_votes = 0
            neg_votes = 0
            # Create an index array and shuffle it
            story_indices = np.arange(len(stories))
            np.random.shuffle(story_indices)
            for story_index in story_indices:
                story_ideology = stories[story_index]
                vote_result = vote(u = user_ideology, s = story_ideology, t_plus = t_p, t_minus = t_m)
                if vote_result == 1 and pos_votes <  deg_seq_pos[user_index]:
                    pos_votes += 1
                    mat[user_index, story_index] = vote_result
                elif vote_result == -1 and neg_votes <  deg_seq_neg[user_index]:
                    neg_votes += 1
                    mat[user_index, story_index] = vote_result
        else:
            for story_index, story_ideology in enumerate(stories):
                mat[user_index, story_index] = vote(u = user_ideology, s = story_ideology, t_plus = t_p, t_minus = t_m)

    return mat




#####################
# DEGREE OF BALANCE #
#####################
'''
# Define a function to calculate the fraction of balanced triangles in a network
def fraction_balanced_triangles(adjacency_matrix):

    # Convert the adjacency matrix to a NumPy array for easier manipulation
    adj_matrix = np.array(adjacency_matrix)
    
    # Get the number of nodes in the network
    num_nodes = len(adj_matrix)
    
    # Initialize a variable to count the balanced triangles
    balanced_triangle_count = 0

    balanced_triangle_count_ppp = 0
    balanced_triangle_count_nnp = 0
    unbalanced_triangle_count_npp = 0
    unbalanced_triangle_count_nnn = 0


    # Initialize a variable to count the triangles
    triangle_count = 0
    
    # Iterate through all possible triangles in the network
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            for k in range(j + 1, num_nodes):
                if adj_matrix[i, j] != 0 and adj_matrix[j, k] != 0 and adj_matrix[i, k] != 0:
                    # This is a triangle
                    triangle_count += 1
                    if adj_matrix[i, j] > 0 and adj_matrix[j, k] > 0 and adj_matrix[i, k] > 0:
                        # This is a balanced triangle
                        balanced_triangle_count += 1
                        balanced_triangle_count_ppp += 1
                    elif adj_matrix[i, j] < 0 and adj_matrix[j, k] < 0 and adj_matrix[i, k] > 0:
                        # This is a balanced triangle
                        balanced_triangle_count += 1
                        balanced_triangle_count_nnp += 1
                    elif adj_matrix[i, j] < 0 and adj_matrix[j, k] > 0 and adj_matrix[i, k] > 0:
                        unbalanced_triangle_count_npp += 1
                    elif adj_matrix[i, j] < 0 and adj_matrix[j, k] < 0 and adj_matrix[i, k] < 0:
                        unbalanced_triangle_count_nnn += 1
                    else:
                        pass
                else:
                    pass

    print(f'no. of triangles = {triangle_count}, balanced triangles = {balanced_triangle_count}, +++ =  {balanced_triangle_count_ppp}, --+ = {balanced_triangle_count_nnp}, -++ = {unbalanced_triangle_count_npp}, --- = {unbalanced_triangle_count_nnn}')
    
    
    if triangle_count > 0:
        # Return the count of balanced triangles
        return balanced_triangle_count/triangle_count, triangle_count, balanced_triangle_count, balanced_triangle_count_ppp, balanced_triangle_count_nnp, unbalanced_triangle_count_npp, unbalanced_triangle_count_nnn
    else:
        return triangle_count, balanced_triangle_count, balanced_triangle_count_ppp, balanced_triangle_count_nnp, unbalanced_triangle_count_npp, unbalanced_triangle_count_nnn
'''
import numpy as np

def fraction_balanced_triangles(adjacency_matrix):
    # Convert the adjacency matrix to a NumPy array for easier manipulation
    adj_matrix = np.array(adjacency_matrix)

    # Calculate the cube of the adjacency matrix
    adj_matrix_cubed = np.linalg.matrix_power(adj_matrix, 3)

    # The number of triangles is half the sum of the diagonal of the cubed adjacency matrix
    triangle_count = np.trace(adj_matrix_cubed) / 6

    # Calculate the number of balanced and unbalanced triangles
    balanced_triangle_count = np.count_nonzero(adj_matrix_cubed > 0) / 6
    unbalanced_triangle_count = triangle_count - balanced_triangle_count

    # Calculate the number of each type of triangle
    balanced_triangle_count_ppp = np.count_nonzero(adj_matrix_cubed == 3) / 6
    balanced_triangle_count_nnp = np.count_nonzero(adj_matrix_cubed == -1) / 6
    unbalanced_triangle_count_npp = np.count_nonzero(adj_matrix_cubed == 1) / 6
    unbalanced_triangle_count_nnn = np.count_nonzero(adj_matrix_cubed == -3) / 6

    print(f'no. of triangles = {triangle_count}, balanced triangles = {balanced_triangle_count}, +++ =  {balanced_triangle_count_ppp}, --+ = {balanced_triangle_count_nnp}, -++ = {unbalanced_triangle_count_npp}, --- = {unbalanced_triangle_count_nnn}')

    if triangle_count > 0:
        # Return the count of balanced triangles
        return balanced_triangle_count/triangle_count, triangle_count, balanced_triangle_count, balanced_triangle_count_ppp, balanced_triangle_count_nnp, unbalanced_triangle_count_npp, unbalanced_triangle_count_nnn
    else:
        return triangle_count, balanced_triangle_count, balanced_triangle_count_ppp, balanced_triangle_count_nnp, unbalanced_triangle_count_npp, unbalanced_triangle_count_nnn

#########
# PLOTS #
#########

# Define a function to generate bins for degree sequences
def bins(deg_seq):
    bins = [min(deg_seq)]
    cur_value = bins[0]
    multiplier = 2
    while cur_value < max(deg_seq):
        cur_value = cur_value * multiplier
        bins.append(cur_value)

    bins = np.array(bins)
    return bins

# Define a function to plot histograms of user and story characteristics
def plot_hist(amplitude, dist_users, dist_stories, sigma, n_users, n_stories, ax=None):
    users, users_labels = gaussian_mix(A=amplitude,
                        mu_a=- dist_stories,
                        mu_b = + dist_stories,sigma_a = sigma,
                        sigma_b = sigma,n_samples = n_users)
    
                        
    stories, stories_labels = gaussian_mix(A=amplitude,
                        mu_a=- dist_users,
                                    mu_b = + dist_users,sigma_a = sigma,
                                    sigma_b = sigma,n_samples = n_stories)


    x = np.linspace(-abs(dist_stories-dist_users)-10*sigma,abs(dist_stories-dist_users)+10*sigma, 10000)

    y = difference_gaussian_mixtures(x,
                                    A = amplitude,
                                    B = amplitude,
                                    mu_1 = - dist_stories,
                                    mu_2 = + dist_stories,
                                    mu_3 = - dist_users,
                                    mu_4 = + dist_users,
                                    sigma_1 = sigma,
                                    sigma_2 = sigma,
                                    sigma_3 = sigma,
                                    sigma_4 = sigma, cdf = False)
    ax.hist(stories, bins = 100, alpha = 0.5, label='stories', density = True, color = '#911D56')

    ax.hist(users, bins = 100,  alpha =0.5, label = 'users', density = True, color = '#DD9562')

    ax.set_yticks([])
        #ax.set_xticks(size = 15)

    #plt.legend(fontsize =15, bbox_to_anchor = (1,1))
    #plt.show()
    return users, users_labels, stories, stories_labels


# Define a function to plot degree distributions of friends and enemies
def plot_deg_distr(A, ax=None):
    deg_plus = np.array(np.sum(A[A>0], axis =0))[0]
    deg_minus = np.array(-np.sum(A[A<0], axis =0))[0]
    if len(deg_plus) > 0:
        ax.hist(deg_plus, color='royalblue', label = 'Friends',alpha = 0.5, bins = bins(deg_plus), density = True)
    if len(deg_minus) > 0:
        ax.hist(deg_minus, color='indianred', label = 'Enemies',alpha = 0.5, bins =bins(deg_minus), density = True)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yticks([])
    #plt.legend()
       








