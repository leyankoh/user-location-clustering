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

# Load Data
# Make histogram
ny = gpd.read_file(os.path.join("Data", "ShapeFiles", "ny_gridID.shp"))
ny = pd.DataFrame(ny) # turn into a dataframe

## Upper hierarchy (9-vc) data set preparation ##
# Merge upper hierarchy
hierarchy = pd.read_csv(os.path.join('Data', 'my_foursquare_tree.csv'), header=None)
hierarchy.columns = ['any','parent', 'subparent', 'venueCat_1']
# merge the two frames based on the lowest hierarchy
merged = pd.merge(ny, hierarchy, on='venueCat_1')
matrix2 = pd.crosstab(index=merged['id'], columns=merged['parent']) # dense matrix

####
# Spectral Clustering
graph2 = cosine_similarity(matrix2) # Graph2 will always be the non-sparse matrix
spec_dense = SpectralClustering(n_clusters=9, affinity='precomputed', n_init=100) # precomputed cosine similarity matrix
dense_clusters = spec_dense.fit_predict(graph2)

# Visualise clusters
matrix2['scluster'] = pd.Series(dense_clusters, index=matrix2.index) # add the cluster results to each grid
matrix2['gridID'] = matrix2.index
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

# Make User histograms
merged_clusters = pd.merge(merged, matrix2[['gridID', 'scluster']], how='left', left_on='id', right_on='gridID')
merged_matrix = pd.crosstab(index=merged_clusters['userId'], columns=merged_clusters['scluster'])

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

## Evaluate clusters of
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(mds)
    cluster_labels = kmeans.fit_predict(mds)
    silhouette_avg = silhouette_score(mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# Distribution
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