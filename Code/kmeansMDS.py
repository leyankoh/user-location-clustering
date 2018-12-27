#### Libraries ####
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
seaborn.set(style='ticks')

from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# load data
nyc = pd.read_csv(os.path.join('Data', 'dataset_TSMC2014_NYC.csv')) # load data
# transform data into matrix
matrix = pd.crosstab(index=nyc['userId'], columns = nyc['venueCategory'])

##################
# Part 1. naive k-means clustering (nyc data only)

# range of clusters for which we want to test
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
kmeans_silhouette = [] # initialise empty list to store groups

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(matrix)
    cluster_labels = kmeans.fit_predict(matrix)
    silhouette_avg = silhouette_score(matrix, cluster_labels)

    kmeans_silhouette.append(silhouette_avg)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# plot the silhouette scores

fig0 = plt.figure()
ax = plt.subplot()
ax.plot(kmeans_silhouette)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
ax.set_xticks(range(0,15))
ax.set_xticklabels(range_n_clusters)
fig0 = plt.gcf()
plt.show()
fig0.savefig(os.path.join('Outputs', 'kmeans_silhouette.png'))
plt.close()

# Plotting distribution of categories from the best splits
chosen_k = [2, 3, 4, 5]
for i in chosen_k:
    kmeans = KMeans(n_clusters = i).fit(matrix)
    label_dist = pd.value_counts(pd.Series(kmeans.labels_))
    label_dist.sort_index(inplace=True)
    fig = plt.figure()
    ax = plt.subplot()
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    label_dist.plot(kind='bar', sort_columns=False)
    fig = plt.gcf()
    plt.show()
    fig.savefig(os.path.join('Outputs', str(i)+ ' ' + 'Clusters.png'))


# Splitting up the users into incredibly small clusters
# their histogram profile will not be the same

k = 200
kmeans = KMeans(n_clusters = k).fit(matrix)
matrix_analysis = matrix.copy(deep=True) # set a deep copy for the e
matrix_analysis['kmeans'] = pd.Series(kmeans.labels_, index = matrix_analysis.index) # add clusters

# check the counts of every group
print pd.value_counts(matrix_analysis.kmeans)
# something like cluster __ has adequate number of users to show that the distributions might differ
filtered = matrix_analysis.loc[matrix_analysis['kmeans'] == 7]
row = filtered.iloc[4]
row.plot(kind="bar")
plt.show()
plt.close()


#########################################
# Naive k-means clustering with upper hierarchy
# load hierarchy file
hierarchy = pd.read_csv(os.path.join('Data', 'my_foursquare_tree.csv'), header=None)
hierarchy.columns = ['any','parent', 'subparent', 'venueCategory']
# merge the two frames based on the lowest hierarchy
merged = pd.merge(nyc, hierarchy, on='venueCategory')
matrix2 = pd.crosstab(index=merged['userId'], columns=merged['parent']) # convert into a matrix (now dense)

# silhouette scores to check separability
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9]
dense_silhouette = [] # initialise empty list to store groups

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters = n_clusters).fit(matrix2)
    cluster_labels = kmeans.fit_predict(matrix2)
    silhouette_avg = silhouette_score(matrix2, cluster_labels)

    dense_silhouette.append(silhouette_avg)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# plot it
fig2 = plt.figure()
ax = plt.subplot()
ax.plot(dense_silhouette)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
ax.set_xticks(range(0,8))
ax.set_xticklabels(range_n_clusters)
fig2 = plt.gcf()
plt.show()
fig2.savefig(os.path.join('Outputs', 'dense_silhouette.png'))
plt.close()

# plot the distribution for the best k, 4
k = 4
kmeans = KMeans(n_clusters=k).fit(matrix2)
k_obj = kmeans.inertia_ / 9  # check objective function
print k_obj

label_dist = pd.value_counts(pd.Series(kmeans.labels_))
label_dist.sort_index(inplace=True)
fig = plt.figure()
ax = plt.subplot()
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
label_dist.plot(kind='bar', sort_columns=False)
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "Dense_Matrix_4_Cluster.png"))

# Evaluate silhouette score of single cluster
# Evaluate silhouette score of one cluster
sil_sample = silhouette_samples(matrix2, kmeans.labels_)
matrix2['klabels'] = kmeans.labels_
matrix2['silhouette'] = sil_sample

for n in range(k):
    cluster = matrix2.loc[matrix2['klabels'] == n]
    sil = cluster['silhouette'].mean()

    print ("Sil average for cluster " + str(n) + ": ", sil)

###############################
# Part 2. Dimensionality reduction with k-means clustering (nyc only)

# use original sparse matrix
# mds api automatically uses euclidean distances
clf = manifold.MDS(n_components=2) # pick mds with just 2 components
X_mds = clf.fit_transform(matrix) # fit matrix into mds

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    kmeans_mds = KMeans(n_clusters = n_clusters).fit(X_mds)
    cluster_labels = kmeans_mds.fit_predict(X_mds)
    silhouette_avg = silhouette_score(X_mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

#plot the scatter
fig3 = plt.figure()
ax = plt.subplot()
plt.scatter(X_mds[:,0], X_mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig3 = plt.gcf()
plt.show()
fig3.savefig(os.path.join("Outputs", "mds_sparse_plot.png"))
plt.close()

# Try to compute a dissimilarity matrix, as the default euclidean distances may prove ineffective
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
cossim = cosine_similarity(matrix)
# convert to dissimilarity
# https://stackoverflow.com/questions/27337610/use-similarity-matrix-instead-of-dissimilarity-matrix-for-mds-in-scikit-learn
# https://math.stackexchange.com/questions/102924/cosine-similarity-distance-and-triangle-equation
dissim = 1 - cossim
clf = manifold.MDS(n_components=2, dissimilarity="precomputed")
dissim_mds = clf.fit_transform(dissim)

# plot

fig = plt.figure()
ax = plt.subplot()
plt.scatter(dissim_mds[:,0], dissim_mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "mds_cosine_sparse_plot.png"))
plt.close()

# now to check the number of clusters in here
range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:
    kmeans_mds = KMeans(n_clusters=n_clusters).fit(dissim_mds)
    cluster_labels = kmeans_mds.fit_predict(dissim_mds)
    silhouette_avg = silhouette_score(dissim_mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)
# 7 groups seem to be the best
##################################################
# dimensionality reduction with non-sparse matrix
clf = manifold.MDS(n_components=2)
dense_mds = clf.fit_transform(matrix2)

range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:
    kmeans_mds = KMeans(n_clusters=n_clusters).fit(dense_mds)
    cluster_labels = kmeans_mds.fit_predict(dense_mds)
    silhouette_avg = silhouette_score(dense_mds, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# colour code it
k = 4
kmeans = KMeans(n_clusters=k).fit(dense_mds)
dense_mds = pd.DataFrame(dense_mds)
dense_mds['kmeans'] = pd.Series(kmeans.labels_, index=dense_mds.index)

# objective function
obj_mds_dir = kmeans.inertia_ / 9
print obj_mds_dir

# plot with colour code
fg = sns.FacetGrid(data=dense_mds, hue='kmeans', aspect=1.61)
fg.map(plt.scatter, 0, 1).add_legend()
fig1 = plt.gcf()
plt.show()
fig1.savefig(os.path.join("Outputs", "mds_dense_4k_plot.png"))
plt.close()


fig4 = plt.figure()
ax = plt.subplot()
plt.scatter(dense_mds[:,0], dense_mds[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig4 = plt.gcf()
plt.show()
fig4.savefig(os.path.join("Outputs", "mds_dense_plot.png"))
plt.close()

# euclidean distance does not seem to be working so well
# try cosine distance as well
dissim2 = cosine_distances(matrix2)

clf = manifold.MDS(n_components=2, dissimilarity="precomputed")
dissim_mds2 = clf.fit_transform(dissim2)
# evaluate silhouette score
range_n_clusters = [2, 3, 4, 5, 6, 7]

for n_clusters in range_n_clusters:
    kmeans_mds = KMeans(n_clusters=n_clusters).fit(dissim_mds2)
    cluster_labels = kmeans_mds.fit_predict(dissim_mds2)
    silhouette_avg = silhouette_score(dissim_mds2, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)

# best k = 4
k = 4
kmeans = KMeans(n_clusters=k).fit(dissim_mds2)

dissim_mds2 = pd.DataFrame(dissim_mds2)
dissim_mds2['kmeans'] = kmeans.labels_
#########################
# Evaluate
# check cluster results
ax = dissim_mds2['kmeans'].value_counts(sort=False).plot.bar()
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
fig.savefig(os.path.join('Outputs', 'mds_cluster_dist2.png'))
plt.close()

# check objectivev function
obj_mds = kmeans.inertia_ / 2
print obj_mds

# check silhouette score of each

# Evaluate silhouette score of each cluster
mds_copy = dissim_mds2.copy(deep=True)
mds_copy.drop(['kmeans'], axis=1, inplace=True)
sil_mds_sample = silhouette_samples(mds_copy, kmeans.labels_)

dissim_mds2['silhouette'] = sil_mds_sample
for n in range(k):
    cluster = dissim_mds2.loc[dissim_mds2['kmeans'] == n]
    sil = cluster['silhouette'].mean()

    print ("Sil average for cluster " + str(n) + ": ", sil)


# plot it!

fig = plt.figure()
ax = plt.subplot()
plt.scatter(dissim_mds2[:,0], dissim_mds2[:,1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
fig = plt.gcf()
plt.show()
fig.savefig(os.path.join("Outputs", "mds_cosine_dense_plot.png"))
plt.close()

# Calculate silhouette score of cosine mds (9-category matrix)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

for n_clusters in range_n_clusters:
    kmeans_mds = KMeans(n_clusters=n_clusters).fit(dissim_mds2)
    cluster_labels = kmeans_mds.fit_predict(dissim_mds2)
    silhouette_avg = silhouette_score(dissim_mds2, cluster_labels)

    print ("The number of clusters is: ", n_clusters,
           "Average score: ", silhouette_avg)
