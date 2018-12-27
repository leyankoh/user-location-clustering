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



###################################
# 1. Spectral Clustering on Sparse Matrix
# get eigengap heuristic
from sklearn.utils.graph import graph_laplacian
from sklearn.utils.arpack import eigsh
from sklearn.manifold.spectral_embedding_ import _set_diag


# try using row percentage
perc_matrix = matrix.copy(deep=True) # set a copy of the matrix to test
perc_matrix['total'] = matrix.sum(axis=1)
percent = perc_matrix.div(perc_matrix['total'], axis='index') * 100 # calculate row percentage
percent = percent.drop(['total'], axis=1) # drop total column

cat_perc = []
for cat in matrix.columns:
    cat_tuple = (cat, matrix[cat].mean())
    cat_perc.append(cat_tuple)
# sort category percentages
cat_perc = sorted(cat_perc, key=lambda x: x[1])

graph = cosine_similarity(matrix) # use cosine similarity, as in Noulas et al.

# https://github.com/mingmingyang/auto_spectral_clustering/blob/master/autosp.py
# how to calculate spectral clusters
norm_laplacian, dd = graph_laplacian(graph, normed=True, return_diag=True)
laplacian = _set_diag(norm_laplacian, 1, norm_laplacian=True)
n_components = graph.shape[0] - 1

eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
eigenvalues = -eigenvalues[::-1]

max_gap = 0
gap_pre_index = 0
for i in range(1, eigenvalues.size):
    gap = eigenvalues[i] - eigenvalues[i-1]
    if gap > max_gap:
        max_gap = gap
        gap_pre_index = i - 1

k = gap_pre_index + 1

print k


# plot eigenvalues to check eigengap
# plot the eigenvalues
plt.plot(eigenvalues[0:15])
plt.xlabel('Gap')
plt.ylabel('Eigenvalue')
plt.xticks(np.arange(15))

fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "Eigengap_sparse_perc.png"))

#####################################
# 1b. Eigengap calculation on frequency location histogram
# Exact same results as using percentage
graph = cosine_similarity(matrix) # use cosine similarity, as in Noulas et al.

norm_laplacian, dd = graph_laplacian(graph, normed=True, return_diag=True)
laplacian = _set_diag(norm_laplacian, 1, norm_laplacian=True)
n_components = graph.shape[0] - 1

eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
eigenvalues = -eigenvalues[::-1]

max_gap = 0
gap_pre_index = 0
for i in range(1, eigenvalues.size):
    gap = eigenvalues[i] - eigenvalues[i-1]
    if gap > max_gap:
        max_gap = gap
        gap_pre_index = i - 1

k = gap_pre_index + 1

print k

# plot eigenvalues to check eigengap
# plot the eigenvalues
plt.plot(eigenvalues[0:15])
plt.xlabel('Gap')
plt.ylabel('Eigenvalue')
plt.xticks(np.arange(15))

fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "Eigengap_sparse.png"))
###################################
# try implementing kneepoint method
from math import atan
def calc_angle(m, m1, m2):
    """
    :param m: current value
    :param m1: next value
    :param m2: previous value
    :return: angle
    """
    angle = atan(1/abs(m-m2)) + atan(1/abs(m1-m))
    return angle

eigenvs = eigenvalues[0:15] # get only first 15 eigenvalues
for n in range(1, 14):
    m = eigenvs[n]
    m1 = eigenvs[n + 1]
    m2 = eigenvs[n - 1]

    angle = calc_angle(m, m1, m2)
    print "Angle of {0}: {1}".format(n, angle)

# get gap values
for n in range(1, 15):
    gap = eigenvalues[n] - eigenvalues[n-1]
    print "{0} gap: {1}".format(n, gap)
####################################
# Use a value of 10 to implement spectral clustering
sc = SpectralClustering(n_clusters=7, affinity='precomputed', n_init=100)
sparse_cluster = sc.fit_predict(graph)

# Evaluate clusters according to distribution
matrix['scluster'] = sparse_cluster


ax = matrix['scluster'].value_counts(sort=False).plot.bar()
plt.xlabel("Cluster Group")
plt.ylabel("Count")
#get plot counts with label on top
# https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
rects = ax.patches
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width()/2

    space = 5 # no of points between bar and label
    va = 'bottom'

    label = "{:.1f}".format(y_value)

    plt.annotate(label, (x_value, y_value), xytext=(0,space), textcoords = 'offset points', ha='center', va=va)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "sc_sparse_grid_clusters_distribution.png"))

# Try to describe spectral clusters
for n in range(7):
    clustering = matrix.loc[matrix['scluster'] == n] # filter out the cluster
    clustering.drop(['scluster', 'id'], axis=1, inplace=True) # remove the cluster and grid ID for more accurate description
    # convert everything to percentage
    clustering['total'] = clustering.sum(axis=1)
    percentages = clustering.div(clustering['total'], axis='index') * 100

    print "Describing... " + str(n)
    print percentages.describe()



########################
# return spectral clusters to get user counts
matrix.reset_index(inplace=True) # reset index
user_cluster_merged = pd.merge(ny, matrix[['id', 'scluster']], how='left', left_on='id', right_on='id')
user_counts = pd.crosstab(index=user_cluster_merged['userId'], columns=user_cluster_merged['scluster'])

# silhouette analysis to check best k
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(user_counts)
    cluster_labels = kmeans.fit_predict(user_counts)
    silhouette_avg = silhouette_score(user_counts, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# use k = 4
kmeans = KMeans(n_clusters=4).fit(user_counts)
obj_function = kmeans.inertia_

####
# Visualise distributions between clusters
user_counts['kcluster'] = kmeans.labels_
ax = user_counts['kcluster'].value_counts(sort=False).plot.bar()
plt.xlabel("Cluster Group")
plt.ylabel("Count")
#get plot counts with label on top
# https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
rects = ax.patches
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width()/2

    space = 5 # no of points between bar and label
    va = 'bottom'

    label = "{:.1f}".format(y_value)

    plt.annotate(label, (x_value, y_value), xytext=(0,space), textcoords = 'offset points', ha='center', va=va)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "lsk_sparse_kmeans_user_counts_4.png"))
####################################
# User k = 6
user_counts.drop(['kcluster'], axis=1, inplace=True)
kmeans = KMeans(n_clusters=6).fit(user_counts)


####
# Visualise distributions between clusters
user_counts['kcluster'] = kmeans.labels_
ax = user_counts['kcluster'].value_counts(sort=False).plot.bar()
plt.xlabel("Cluster Group")
plt.ylabel("Count")
#get plot counts with label on top
# https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
rects = ax.patches
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width()/2

    space = 5 # no of points between bar and label
    va = 'bottom'

    label = "{:.1f}".format(y_value)

    plt.annotate(label, (x_value, y_value), xytext=(0,space), textcoords = 'offset points', ha='center', va=va)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "lsk_sparse_kmeans_user_counts_6.png"))

print kmeans.inertia_

####################################
# Try mds on sparse counts
user_counts.drop(['kcluster'], axis=1, inplace=True)
user_affinity = cosine_distances(user_counts)
clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
ny_mds = clf.fit_transform(user_affinity)

plt.scatter(ny_mds[:,0], ny_mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "lsmk_sparse_mds.png"))

# Check silhouette scores
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(ny_mds)
    cluster_labels = kmeans.fit_predict(ny_mds)
    silhouette_avg = silhouette_score(ny_mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

###################################
# 2. Spectral Clustering on Non-Sparse Matrix (Location venues) TBC
graph2 = cosine_similarity(matrix2) # Graph2 will always be the non-sparse matrix

norm_laplacian, dd = graph_laplacian(graph2, normed=True, return_diag=True) # calculate normalised laplacian of affinity matrix
laplacian = _set_diag(norm_laplacian, 1, norm_laplacian=True) # not sure what this does, possible set diagonal
n_components = graph2.shape[0] - 1


# calculate the eigenvalues and eigenvectors of the normalized laplacian
eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
eigenvalues = -eigenvalues[::-1] # invert (including sign).

max_gap = 0
gap_pre_index = 0
for i in range(1, eigenvalues.size):
    gap = eigenvalues[i] - eigenvalues[i-1]
    if gap > max_gap:
        max_gap = gap
        gap_pre_index = i - 1

k = gap_pre_index + 1

print k

# plot the eigenvalues
plt.plot(eigenvalues[0:15])
plt.xlabel('Gap')
plt.ylabel('Eigenvalue')
plt.xticks(np.arange(15))

fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "Eigengap_dense.png"))


############################
# Carry out spectral clustering with 9 clusters as k=9
#n_init = Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
spec_dense = SpectralClustering(n_clusters=9, affinity='precomputed', n_init=100) # precomputed cosine similarity matrix
dense_clusters = spec_dense.fit_predict(graph2)

matrix2['scluster'] = pd.Series(dense_clusters, index=matrix2.index) # add the cluster results to each grid
matrix2['gridID'] = matrix2.index

# plotting the distribution shows that the clusters are well distributed
ax = matrix2['scluster'].value_counts(sort=False).plot.bar()
plt.xlabel("Cluster Group")
plt.ylabel("Count")
#get plot counts with label on top
# https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
rects = ax.patches
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width()/2

    space = 5 # no of points between bar and label
    va = 'bottom'

    label = "{:.1f}".format(y_value)

    plt.annotate(label, (x_value, y_value), xytext=(0,space), textcoords = 'offset points', ha='center', va=va)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "sc_dense_grid_clusters_distribution.png"))

# save the clustering so that I don't have to keep re-running the code
matrix2.to_csv(os.path.join('Data', 'sc_dense_grid_cluster_data.csv'))
# Load csv file here
matrix2 = pd.read_csv(os.path.join('Data', 'sc_dense_grid_cluster_data.csv'))
matrix2.drop(['id'], axis=1, inplace=True)
# analyse for cluster distribution
# use percentages to describe clusters as it is more intuitive
for n in range(9):
    clustering = matrix2.loc[matrix2['scluster'] == n] # filter out the cluster
    clustering.drop(['scluster', 'gridID'], axis=1, inplace=True) # remove the cluster and grid ID for more accurate description
    # convert everything to percentage
    clustering['total'] = clustering.sum(axis=1)
    percentages = clustering.div(clustering['total'], axis='index') * 100

    print "Describing... " + str(n)
    print percentages.describe()

#join the clusters back up with the grid id so that we know the
# clustering of the location each user checked into
# match check-ins to ID
# column = cluster
# id = user id
# rows = number of times a user checked into that cluster

merged_clusters = pd.merge(merged, matrix2[['gridID', 'scluster']], how='left', left_on='id', right_on='gridID')
merged_matrix = pd.crosstab(index=merged_clusters['userId'], columns=merged_clusters['scluster'])

# save the user location histogram (no repeat)
merged_matrix.to_csv(os.path.join("Data", "dense_matrix_user_histogram.csv"))

# load
merged_matrix = pd.read_csv(os.path.join('Data', 'dense_matrix_user_histogram.csv'))
merged_matrix.drop(['userId'], axis=1, inplace=True)
# now to apply kmeans to it
# check silhouette scores
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(merged_matrix)
    cluster_labels = kmeans.fit_predict(merged_matrix)
    silhouette_avg = silhouette_score(merged_matrix, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)
# best score is 5
k = 5
kmeans = KMeans(n_clusters = k).fit(merged_matrix) # fit kmeans onto the matrix
merged_matrix['klabels'] = pd.Series(kmeans.labels_, index=merged_matrix.index)

# plot distribution of users
# it seems the distribution is incredibly poor
ax = merged_matrix['klabels'].value_counts(sort=False).plot.bar()
plt.xlabel('Cluster Group')
plt.ylabel('Count')
rects = ax.patches
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width()/2

    space = 5 # no of points between bar and label
    va = 'bottom'

    label = "{:.1f}".format(y_value)

    plt.annotate(label, (x_value, y_value), xytext=(0,space), textcoords = 'offset points', ha='center', va=va)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join('Outputs', 'Spectral_Kmeans_cluster_dist.png'))

###########################
# Evaluate the objective function of k-means
obj_func = kmeans.inertia_

# Evaluate silhouette score of one cluster
merged_matr_copy = merged_matrix.copy(deep=True)
merged_matr_copy.drop(['klabels'], axis=1, inplace=True) # need to delete or will affect score
sil_sample = silhouette_samples(merged_matr_copy, kmeans.labels_)

merged_matrix['silhouette'] = sil_sample # add silhouette score of each sample as a column
for n in range(5):
    cluster = merged_matrix.loc[merged_matrix['klabels'] == n]
    sil = cluster['silhouette'].mean() # calculate mean of all silhouette

    print ("Sil average for cluster " + str(n) + ": ", sil)

##############################################
# Testing mds for user histograms
# perform mds so we can see the layout
user_dif = cosine_distances(merged_matrix)

clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds = clf.fit_transform(user_dif)

# plot the mds
fig = plt.figure()
ax = plt.subplot()
plt.scatter(mds[:,0], mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "user_histogram_dense_mds.png"))
plt.close()

# check silhouette scores of mds
# it actually performs worse than just outright doing k-means
# best cluster k = 6
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(mds)
    cluster_labels = kmeans.fit_predict(mds)
    silhouette_avg = silhouette_score(mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)
######################################
# 2b. Evaluate Lsmk
k = 6
kmeans = KMeans(n_clusters=k).fit(mds)

# add cluster results
merged_matrix['mds_cluster'] = kmeans.labels_
# check cluster balance
ax = merged_matrix['mds_cluster'].value_counts(sort=False).plot.bar()
plt.xlabel('Cluster Group')
plt.ylabel('Count')
rects = ax.patches
for rect in rects:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width()/2

    space = 5 # no of points between bar and label
    va = 'bottom'

    label = "{:.1f}".format(y_value)

    plt.annotate(label, (x_value, y_value), xytext=(0,space), textcoords = 'offset points', ha='center', va=va)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join('Outputs', 'lsmk_mds_cluster_dist.png'))
plt.close()

# visualise mds on chart
mds = pd.DataFrame(mds)
mds['mds_labels'] = kmeans.labels_


fg = sns.FacetGrid(data=mds, hue='mds_labels')
fg.map(plt.scatter, 0, 1).add_legend()
plt.xlabel('First Component')
plt.ylabel('Second Component')
fig = plt.gcf()
fig.set_size_inches(8,8)
plt.show()
fig.savefig(os.path.join('Outputs', 'lsmk_mds_cluster_labels.png'))

# check objective function
obj_mds = kmeans.inertia_ / 2
print obj_mds

# Evaluate silhouette score of each cluster
mds_copy = mds.copy(deep=True)
mds_copy.drop(['mds_labels'], axis=1, inplace=True)
sil_mds_sample = silhouette_samples(mds_copy, kmeans.labels_)

mds['silhouette'] = sil_mds_sample
for n in range(k):
    cluster = mds.loc[mds['mds_labels'] == n]
    sil = cluster['silhouette'].mean()

    print ("Sil average for cluster " + str(n) + ": ", sil)


# Evaluate the features of each user
for n in range(6):
    clustering = merged_matrix.loc[merged_matrix['mds_cluster'] == n] # filter clsuter out
    clustering.drop(['mds_cluster'], axis=1, inplace=True)

    clustering['total'] = clustering.sum(axis=1) # sum over all columns
    perc = clustering.div(clustering['total'], axis='index') * 100

    print '=' * 20
    print "Describing... " + str(n)
    print '=' * 20
    print perc.describe()

merged_matrix.to_csv(os.path.join('Data', 'mds_clusters_lsmk.csv')) # in case i need to access the clusters later

mds_cluster = pd.read_csv(os.path.join('Data', 'mds_clusters_lsmk.csv')) # reload to save the state
mds_cluster.drop(['Unnamed: 0'], axis=1, inplace=True)

# Evaluate shape of histograms
cluster4 = mds_cluster.copy(deep=True).loc[mds_cluster['mds_cluster'] == 4] # filter 4th cluster only
cluster4.drop(['mds_cluster'], axis=1, inplace=True)
cluster4.reset_index(inplace=True)
cluster_transposed = cluster4.set_index('index').T

fig, ax = plt.subplots()
ax = cluster_transposed.plot.hist(alpha=0.5)
plt.show()
plt.close()

cluster4.drop(['index'], axis=1, inplace=True)
fig, ax = plt.subplots()
ax = cluster4.plot.hist(alpha=0.5)
plt.show()
plt.close()

#for i, data in cluster4.iterrows():
#    ax.hist(data[''])
######################################
# 3. k-Means + MDS on sparse matrix (location venues)
# test only, irrelevant to the thesis
cossim = cosine_similarity(matrix)
# create dissimilarity matrix for mds
dissim = 1 - cossim

clf = manifold.MDS(n_components=2, dissimilarity='precomputed')
sparse_mds = clf.fit_transform(dissim)
clf = manifold.MDS(n_components=2)
X_mds = clf.fit_transform(matrix)

#quickly save because it took so long
np.save(os.path.join('Outputs', 'sparse_mds_location_cluster'), sparse_mds) # for mds performed with cosine dissim
np.save(os.path.join('Outputs', 'sparse_location_cluster'), X_mds) # euclidean distance


# plot cosine dissimilarity mds
fig4 = plt.figure()
ax = plt.subplot()
plt.scatter(sparse_mds[:,0], sparse_mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig4 = plt.gcf()
plt.show()
fig4.savefig(os.path.join("Outputs", "location_mds_cosine_sparse_plot.png"))
plt.close()

# this is just the mds without cosine dissimilarity - not too good results
fig4 = plt.figure()
ax = plt.subplot()
plt.scatter(X_mds[:,0], X_mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig4 = plt.gcf()
plt.show()
fig4.savefig(os.path.join("Outputs", "location_mds_sparse_plot.png"))
plt.close()

#########################################
# 4. Assign clusters to k-Means + MDS Location venues

cosine_mds = np.load(os.path.join("Outputs", "sparse_mds_location_cluster.npy"))

# check number of appropriate clusters

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans_mds = KMeans(n_clusters = n_clusters).fit(cosine_mds)
    cluster_labels = kmeans_mds.fit_predict(cosine_mds)
    silhouette_avg = silhouette_score(cosine_mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# best cluster is 7

k = 7
kmeans = KMeans(n_clusters = k).fit(cosine_mds)
cosine_mds = pd.DataFrame(cosine_mds) # convert to dataframe so we can get series

########################################
# 5. Describe 9-venue category locations by percentage (to check results)
# try using row percentage
perc_matrix = matrix2.copy(deep=True) # set a copy of the matrix to test
perc_matrix['total'] = matrix2.sum(axis=1)
percent = perc_matrix.div(perc_matrix['total'], axis='index') * 100 # calculate row percentage
percent = percent.drop(['total'], axis=1) # drop total column


graph = cosine_similarity(percent) # use cosine similarity, as in Noulas et al.

# https://github.com/mingmingyang/auto_spectral_clustering/blob/master/autosp.py
# how to calculate spectral clusters
norm_laplacian, dd = graph_laplacian(graph, normed=True, return_diag=True)
laplacian = _set_diag(norm_laplacian, 1, norm_laplacian=True)
n_components = graph.shape[0] - 1

eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
eigenvalues = -eigenvalues[::-1]

max_gap = 0
gap_pre_index = 0
for i in range(1, eigenvalues.size):
    gap = eigenvalues[i] - eigenvalues[i-1]
    if gap > max_gap:
        max_gap = gap
        gap_pre_index = i - 1

k = gap_pre_index + 1

print k

######################################################
# Attempt spectral clustering on user histograms (test)
dense_user_hist = pd.read_csv(os.path.join("Data", "dense_matrix_user_histogram.csv"))
# use user id column as index
dense_user_hist.set_index("userId", inplace=True)

# calculate eigengap
# returns 1
graph = cosine_similarity(dense_user_hist)
norm_laplacian, dd = graph_laplacian(graph, normed=True, return_diag=True) # calculate normalised laplacian of affinity matrix
laplacian = _set_diag(norm_laplacian, 1, norm_laplacian=True) # not sure what this does, possible set diagonal
n_components = graph.shape[0] - 1


# calculate the eigenvalues and eigenvectors of the normalized laplacian
eigenvalues, eigenvectors = eigsh(-laplacian, k=n_components, which="LM", sigma=1.0, maxiter=5000)
eigenvalues = -eigenvalues[::-1] # invert (including sign).

max_gap = 0
gap_pre_index = 0
for i in range(1, eigenvalues.size):
    gap = eigenvalues[i] - eigenvalues[i-1]
    if gap > max_gap:
        max_gap = gap
        gap_pre_index = i - 1

k = gap_pre_index + 1

print k

# visualise 1-15th eigenvectors
plt.plot(eigenvalues[0:15])
plt.xlabel('Gap')
plt.ylabel('Eigenvalue')
plt.xticks(np.arange(15))

fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "Eigengap_kmeans_dense.png"))


sc_users = SpectralClustering(n_clusters=5, affinity='precomputed', n_init=100)
sc_user_clusters = sc_users.fit_predict(graph)

dense_user_hist['scluster'] = pd.Series(sc_user_clusters, index=dense_user_hist.index)

ax = dense_user_hist['scluster'].value_counts(sort=False).plot.bar()
plt.show()

matrix2['scluster'] = pd.Series(dense_clusters, index=matrix2.index) # add the cluster results to each grid
matrix2['gridID'] = matrix2.index

# plotting the distribution shows that the clusters are well distributed
ax = matrix2['scluster'].value_counts(sort=False).plot.bar()
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "sc_dense_grid_clusters_distribution.png"))
