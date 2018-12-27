### Libraries ###
import os
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn import manifold
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import geopandas as gpd
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding('utf8')

################
# Load files (area)
# Load data
ny = gpd.read_file(os.path.join("Data", "ShapeFiles", "ny_gridID.shp"))
ny = pd.DataFrame(ny) # turn into a dataframe
matrix = pd.crosstab(index = ny['id'], columns = ny['venueCat_1']) # make sparse matrix

## Upper hierarchy (9-vc) data set preparation ##
# Merge upper hierarchy
hierarchy = pd.read_csv(os.path.join('Data', 'my_foursquare_tree.csv'), header=None)
hierarchy.columns = ['any','parent', 'subparent', 'venueCat_1']
# merge the two frames based on the lowest hierarchy
merged = pd.merge(ny, hierarchy, on='venueCat_1')
matrix2 = pd.crosstab(index=merged['id'], columns=merged['parent']) # dense matrix

# User frequency
# load data
nyc = pd.read_csv(os.path.join('Data', 'dataset_TSMC2014_NYC.csv')) # load data
# transform data into matrix
freq_matrix = pd.crosstab(index=nyc['userId'], columns = nyc['venueCategory'])


# load hierarchy file
hierarchy = pd.read_csv(os.path.join('Data', 'my_foursquare_tree.csv'), header=None)
hierarchy.columns = ['any','parent', 'subparent', 'venueCategory']
# merge the two frames based on the lowest hierarchy
merged = pd.merge(nyc, hierarchy, on='venueCategory')
freq_matrix_dense = pd.crosstab(index=merged['userId'], columns=merged['parent']) # convert into a matrix (now dense)

##############
# Runtime for k-means histograms
# 251 category
%timeit kmeans = KMeans(n_clusters = 5).fit(freq_matrix)
# 9 category
%timeit kmeans2 = KMeans(n_clusters = 4).fit(freq_matrix_dense)


#############
# Runtime for mds + kmeans
%%timeit
dif = cosine_distances(freq_matrix)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(dif)
kmeans = KMeans(n_clusters = 5).fit(mds)

%%timeit
dif2 = cosine_distances(freq_matrix_dense)
clf = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
mds = clf.fit_transform(dif2)
kmeans = KMeans(n_clusters=4).fit(mds)

############
# Runtime for new algorithm (251 category)
%%timeit
graph = cosine_similarity(matrix)
sc = SpectralClustering(n_clusters=7, affinity='precomputed', n_init=100)
sparse_cluster = sc.fit_predict(graph)

# merge and convert to user frequencies
matrix['scluster'] = sparse_cluster
matrix.reset_index(inplace=True) # reset index
user_cluster_merged = pd.merge(ny, matrix[['id', 'scluster']], how='left', left_on='id', right_on='id')
user_counts = pd.crosstab(index=user_cluster_merged['userId'], columns=user_cluster_merged['scluster'])

# Run new k-means
%timeit kmeans = KMeans(n_clusters=4).fit(user_counts)

# runtime for mds + kmeans
%%timeit
user_affinity = cosine_distances(user_counts)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
ny_mds = clf.fit_transform(user_affinity)
kmeans = KMeans(n_clusters=8).fit(ny_mds)

#Runtime for new algorithm (9 category)
%%timeit
graph2 = cosine_similarity(matrix2)
sc2 = SpectralClustering(n_clusters = 9, affinity='precomputed', n_init=100)
dense_cluster = sc2.fit_predict(graph2)

# merge and convert to user frequencies
matrix2['scluster'] = dense_cluster
matrix2['gridID'] = matrix2.index
merged_clusters = pd.merge(merged, matrix2[['gridID', 'scluster']], how='left', left_on='id', right_on='gridID')
merged_matrix = pd.crosstab(index=merged_clusters['userId'], columns=merged_clusters['scluster'])

# run new k-means (9-cat)
%timeit kmeans = KMeans(n_clusters = 5).fit(merged_matrix)

# run new mds + kmeans
%%timeit
user_dissim = cosine_distances(merged_matrix)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
user_dense_mds = clf.fit_transform(user_dissim)
kmeans = KMeans(n_clusters = 6).fit(user_dense_mds)

##############################
# Runtime with dataset size
# Sample users only
# 9 venue category LMSK

sample = merged_matrix.sample(n=100)
%%timeit
dissim = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(dissim)
kmeans = KMeans(n_clusters=6).fit(mds)


sample = merged_matrix.sample(n=300)
%%timeit
dissim = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(dissim)
kmeans = KMeans(n_clusters=6).fit(mds)


sample = merged_matrix.sample(n=500)
%%timeit
dissim = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(dissim)
kmeans = KMeans(n_clusters=6).fit(mds)


sample = merged_matrix.sample(n=700)
%%timeit
dissim = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(dissim)
kmeans = KMeans(n_clusters=6).fit(mds)


sample = merged_matrix.sample(n=900)
%%timeit
dissim = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(dissim)
kmeans = KMeans(n_clusters=6).fit(mds)
######################
# Test mds + kmeans (9-cat)

sample = freq_matrix_dense.sample(n=100)
%%timeit
dif2 = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
mds = clf.fit_transform(dif2)
kmeans = KMeans(n_clusters=4).fit(mds)

sample = freq_matrix_dense.sample(n=300)
%%timeit
dif2 = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
mds = clf.fit_transform(dif2)
kmeans = KMeans(n_clusters=4).fit(mds)

sample = freq_matrix_dense.sample(n=500)
%%timeit
dif2 = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
mds = clf.fit_transform(dif2)
kmeans = KMeans(n_clusters=4).fit(mds)

sample = freq_matrix_dense.sample(n=700)
%%timeit
dif2 = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
mds = clf.fit_transform(dif2)
kmeans = KMeans(n_clusters=4).fit(mds)

sample = freq_matrix_dense.sample(n=900)
%%timeit
dif2 = cosine_distances(sample)
clf = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
mds = clf.fit_transform(dif2)
kmeans = KMeans(n_clusters=4).fit(mds)

######################
# test lsk k-means (9-cat)

sample = merged_matrix.sample(n=100)
%%timeit
kmeans = KMeans(n_clusters = 5).fit(sample)


sample = merged_matrix.sample(n=300)
%%timeit
kmeans = KMeans(n_clusters = 5).fit(sample)


sample = merged_matrix.sample(n=500)
%%timeit
kmeans = KMeans(n_clusters = 5).fit(sample)


sample = merged_matrix.sample(n=700)
%%timeit
kmeans = KMeans(n_clusters = 5).fit(sample)


sample = merged_matrix.sample(n=900)
%%timeit
kmeans = KMeans(n_clusters = 5).fit(sample)


####################
# test kmeans 9-cat

sample = freq_matrix_dense.sample(n=100)
%timeit kmeans2 = KMeans(n_clusters = 4).fit(sample)

sample = freq_matrix_dense.sample(n=300)
%timeit kmeans2 = KMeans(n_clusters = 4).fit(sample)

sample = freq_matrix_dense.sample(n=500)
%timeit kmeans2 = KMeans(n_clusters = 4).fit(sample)

sample = freq_matrix_dense.sample(n=700)
%timeit kmeans2 = KMeans(n_clusters = 4).fit(sample)

sample = freq_matrix_dense.sample(n=900)
%timeit kmeans2 = KMeans(n_clusters = 4).fit(sample)

##########
# Lsk vs kmeans
lsk = [18.4, 27.2, 32.8, 38.9, 49.1] # time in ms
km = [18.4, 26.9, 27.2, 41.7, 46.5]
ticks = [100, 300, 500, 700, 900]

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(ticks, lsk, label='Lsk Algorithm')
ax.plot(ticks, km, label = 'Direct k-Means')
ax.set_xticks(np.arange(100, 1100, 200))
plt.xlabel("Number of users")
plt.ylabel("Time to run algorithm (ms)")
plt.legend()
fig=plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "lsk_kmeans_runtime_comparison.png"))

###########
# Lsmk vs mds
lsmk = [0.169, 1.06, 5.52, 13.5, 24.3]
mdstime = [0.148, 1.0, 5.89, 12.7, 21.9]

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(ticks, lsmk, label='Lsmk Algorithm')
ax.plot(ticks, mdstime, label = 'Direct MDS')
ax.set_xticks(np.arange(100, 1100, 200))
plt.xlabel("Number of users")
plt.ylabel("Time to run algorithm (s)")
plt.legend()
fig=plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "lsmk_mds_runtime_comparison.png"))


###############
# time according to no. of k
# lsk
%timeit kmeans = KMeans(n_clusters = 2).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 3).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 4).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 5).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 6).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 7).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 8).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 9).fit(merged_matrix)
%timeit kmeans = KMeans(n_clusters = 10).fit(merged_matrix)

# lsmk
%timeit kmeans = KMeans(n_clusters = 2).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 3).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 4).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 5).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 6).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 7).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 8).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 9).fit(user_dense_mds)
%timeit kmeans = KMeans(n_clusters = 10).fit(user_dense_mds)

klsk = [18.5, 31.9, 39.9, 47.8, 53.2, 59.4, 64.2, 73.9, 80.0]
klsmk = [29.6, 52.3, 53.1, 62.7, 66.7, 73.5, 81.2, 82.4, 86.5]

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(2,11), klsk, label='Lsk Algorithm')
ax.plot(np.arange(2,11), klsmk, label = 'Lsmk Algorithm')
ax.set_xticks(np.arange(2, 11))
plt.xlabel("Number of k")
plt.ylabel("Time to run algorithm (ms)")
plt.legend()
fig=plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "lsmk_lsk_k_comparison.png"))