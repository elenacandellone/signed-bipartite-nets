# Import necessary libraries
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import bz2
import pickle

# Define the path to the data directory
path = './data/meneame/real-data/'

# Read votes data from a compressed file
df = pd.read_csv(path + 'df_stories_votes.tsv.gz', sep="\t", compression="gzip").drop_duplicates().reset_index(drop=True)

# Numerical encoding for 'story_id' and 'username_vote'
df["story_index"] = LabelEncoder().fit_transform(df["story_id"])
df["username_index"] = LabelEncoder().fit_transform(df["username_vote"])

# Create an incidence matrix for user-story interactions
matrix = csr_matrix((df["story_vote"].values, (df["story_index"].values, df["username_index"].values)))

# Perform bipartite projection to create the co-voting matrix
A = matrix.T @ (matrix)

# Save the co-voting matrix to a compressed pickle file
with bz2.BZ2File(path+'adj_data.pkl', 'w') as f:
    pickle.dump(A, f)
