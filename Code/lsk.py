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

########
# Load data
ny = gpd.read_file(os.path.join("Data", "ShapeFiles", "ny_gridID.shp"))
ny = pd.DataFrame(ny) # turn into a dataframe
matrix = pd.crosstab(index = ny['id'], columns = ny['venueCat_1']) # make sparse matrix

########
# Spectral Clustering
graph = cosine_similarity(matrix) # use cosine similarity, as in Noulas et al.
sc = SpectralClustering(n_clusters=7, affinity='precomputed', n_init=100)
sparse_cluster = sc.fit_predict(graph)
matrix['scluster'] = sparse_cluster

# Graph distribution of cluster
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

###########
# Get user histogram
matrix.reset_index(inplace=True) # reset index
user_cluster_merged = pd.merge(ny, matrix[['id', 'scluster']], how='left', left_on='id', right_on='id')
user_counts = pd.crosstab(index=user_cluster_merged['userId'], columns=user_cluster_merged['scluster'])

###########
# Silhouette score for best k
# silhouette analysis to check best k
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(user_counts)
    cluster_labels = kmeans.fit_predict(user_counts)
    silhouette_avg = silhouette_score(user_counts, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

kmeans = KMeans(n_clusters=4).fit(user_counts)

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