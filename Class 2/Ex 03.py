# Data Engineering
# NYU Tandon School of Engineering
# Class 2.1 - Grouping data
# Fall 2023
# Prof. Carlos De Oliveira - Carlos.DeOliveira@NYU.edu 

#################################

# Aquisition of new clients, as an example 

#################################

# Clustering: Group similar things together, while separating dissimilar things.

# For numerical data in Euclidean space, the most common way to identify 
# the location of a cluster is by finding the mean of its points, 
# which corresponds to its center of mass. This point is called the CENTROID. 
# It is the geometrical center, and is found with a straightforward mean calculation.

# For data in non-Euclidean space, the story gets a bit more complicated. 
# Examples of non-Euclidean space are string comparison features, 
# and mixed data that has both categorical and numerical data. 
# In these cases, a geometrical center does not exist so we need a different strategy. 
# The most common solution is to identify a MEDIOID in each cluster. 
# A MEDIOID is the data point that is closest to the other points in the cluster. 
# It must be one of the actual data points in the set, 
# and can be thought of as the best representative of that cluster. 

# How do you define "closest"? It is commonly found by looking for 
# the lowest score on one of the following metrics:
#   1- The maximum distance to other points in cluster
#   2- The mean distance to all other points in cluster
#   3- The sum-of-squares distance to all other points

# Similarity
# See PPT

# Termination
# See PPT


#################################

# Clustering example 

#################################

# Clustering: Let's build a function 
# that creates a demonstration dataset of blobs for clustering examples. 

# import datasets module from Sci-kit learn
from sklearn import datasets
# import other libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# function to create data for clustering examples
def make_blobs():
    # build blobs for demonstration
    n_samples = 1500
    blobs = datasets.make_blobs(n_samples=n_samples,
                                 centers=5,
                                 cluster_std=[3.0, 0.9, 1.9, 1.9, 1.3],
                                 random_state=43)

    # create a Pandas dataframe for the data
    df = pd.DataFrame(blobs[0], columns=['Feature_1', 'Feature_2'])
    df.index.name = 'record'
    return df

# generate blob example dataset
df = make_blobs()
print(df.head())

# plot scatter of blob set
sns.lmplot(x='Feature_2', y='Feature_1', 
           data=df, fit_reg=False)


# K-means
# 1. Pick the k initial cluster centers at random from points in the input data
# 2. Assign all the data points to the cluster to which they are closest to
# 3. Move the k centroids to the center of all the points inside the newly created clusters
# 4. Repeat until k clusters stop changing (for example, convergence)

# This method uses centroids to define location, Euclidean distance as similarity metric, 
# and cohesion as the quality score. Termination occurs when the quality score converges, 
# as measured by a change to less than the tolerance amount. The K-means++ and
# mini batch variants in the K-means family are introduced later in the section. 

# re-generate blob example dataset
df = make_blobs()

# import module and instantiate K-means object
from sklearn.cluster import KMeans
clus = KMeans(n_clusters=5, tol=0.004, max_iter=300)

# fit to input data
clus.fit(df)

# get cluster assignments of input data and print first five labels
df['K-means Cluster Labels'] = clus.labels_
print(df['K-means Cluster Labels'][:5].tolist())

# Now, let's use Seaborn's scatter plot to visualize 
# the grouping of a blob set with the cluster labels displayed
sns.lmplot(x='Feature_2', y='Feature_1', 
           hue="K-means Cluster Labels", data=df, fit_reg=False)

# finding k with silhouette
# generate blob example dataset
df = make_blobs()

# find best value for k using silhouette score
# import metrics module
from sklearn import metrics

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df)
    cluster_labels = kmeans.predict(df)
    S = metrics.silhouette_score(df, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(k, S))
# 5 clusters will be our choice

# k-means++
# instantiate k-means object with k-means++ init method
clus = KMeans(n_clusters=5, init='k-means++',
              tol=0.004, max_iter=300)

# k-means mini batch
# import module and instantiate k-means mini batch object
from sklearn.cluster import MiniBatchKMeans
clus = MiniBatchKMeans(n_clusters=5, batch_size=50,
                        tol=0.004, max_iter=300)


# Hierarchical clustering and dendogram
# The goal of hierarchical clustering is to merge similar clusters in a hierarchical fashion. 
# The plot of clusters connected in a hierarchical fashion is called a dendrogram.
# So, if you consider all available clusters as candidates for merging, 
# the two with the lowest pairwise linkage will be chosen for merger. 

# The hierarchical clustering analysis algorithm (HCA) can be written in two different ways. 
# The first is in the agglomerative fashion, which starts with every data point being in its own cluster, 
# then moving up and merging all the way to a single-cluster hierarchy. 
# The second way is divisive in nature and starts with all data points assigned 
# to a single huge cluster, then moves in the opposite direction. 
# Agglomerative clustering is much more common in data mining. 

# An example of applying HCA in Scikit-learn is included in the following code:
# generate blob example dataset
df = make_blobs()

# import module and instantiate HCA object
from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters=5, 
                               affinity='euclidean', linkage='ward')
# fit to input data
clus.fit(df)

# get cluster assignments
df['HCA Cluster Labels'] = clus.labels_

sns.lmplot(x='Feature_2', y='Feature_1', 
           hue="HCA Cluster Labels", data=df, fit_reg=False)

# You only have to build the dendrogram once during an analysis. 
# You can change the level of hierarchyused in the algorithm by simply moving 
# the distance cutoff (the dotted line in our dendrogram) up and down. 
# Since the level of hierarchy controls thenumber of clusters, 
# you can use this to tune the quality of your clustering.
# The following is an example code for reusing the dendrogram and fitting multiple times:
# find optimal number of clusters using silhouette score

# import metrics module and plot dendogram 
from sklearn import metrics

# generate blob example dataset
df = make_blobs()

# import module and instantiate HCA object
from sklearn.cluster import AgglomerativeClustering

# create list of k values to test and then use for loop
n_clusters = [2,3,4,5,6,7,8]
for num in n_clusters:
    HCA = AgglomerativeClustering(n_clusters=num, 
                               affinity='euclidean', linkage='ward',
                               memory='./model_storage/dendrogram', 
                               compute_full_tree=True)
    cluster_labels= HCA.fit_predict(df)
    S = metrics.silhouette_score(df, cluster_labels)
    print("n_clusters = {:d}, silhouette score {:1f}".format(num, S))
# breack here

# plot
# import scipy module
from scipy.cluster import hierarchy

# generate blob example dataset
df = make_blobs()

# Calculate the distance between each sample
Z = hierarchy.linkage(df, 'ward') 

# Plot with Custom leaves (scroll down in console to see plot)
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=8, labels=df.index)

# Density clustering
# As opposed to defining similarity as solely a measure of distance between points, 
# density clustering adds a correction for space covered by those points. 
# After this correction, the number of points in a given space matters when defining clusters. 
# As a consequence density clustering is very good at denoising, 
# which means to exclude noisy outlier points when they lie outside the dense areas of the data. 
# This clustering method also does not require you to know the number of clusters before you run the fit routine.
# The most popular density clustering algorithm is called DBSCAN, 
# and uses the cohesion concept to restrict the definition of density to include only data points within the cluster. 

# An example of applying DBSCAN in Scikit-learn is included in the following code:
# generate blob example dataset
df = make_blobs()

# import module and instantiate DBSCAN object
from sklearn.cluster import DBSCAN
clus = DBSCAN(eps=0.9, min_samples=5, metric='euclidean')

# fit to input data
clus.fit(df)

# get cluster assignments
df['DBSCAN Cluster Labels'] = clus.labels_

sns.lmplot(x='Feature_2', y='Feature_1', 
           hue="DBSCAN Cluster Labels", data=df, fit_reg=False)

# Spectral analysis
# Spectral clustering builds a connection graph and groups points based on 
# the connectivity of its constituent nodes. 
# Unlike density clustering, you do have to know the number of clusters at fit time. 
# A similarity matrix is built that compares the affinity of each data point to the rest of the points. 
# Then, similar to the principal component analysis introduced in the previous chapter, 
# eigenvectors are found, and the data is transformed into this new affinity space. 
# Finally, a conventional clustering algorithm, such as K-means, is used to cluster the data in affinity space. 

# An example of applying spectral clustering in Scikit-learn is included in the following code:
# generate blob example dataset
df = make_blobs()

# import module and instantiate spectral clustering object
from sklearn.cluster import SpectralClustering
clus = SpectralClustering(n_clusters=5, random_state=42, 
                          assign_labels='kmeans', n_init=10,
                          affinity='nearest_neighbors', n_neighbors=10)
# fit to input data
clus.fit(df)

# get cluster assignments
df['Spectral Cluster Labels'] = clus.labels_

sns.lmplot(x='Feature_2', y='Feature_1', 
           hue="Spectral Cluster Labels", data=df, fit_reg=False)


#################################

# Prediction with regression and classification

#################################

# Topics:
# Mathematical machinery, including loss functions and gradient descent
# Linear regression and penalties
# Logistic regression
# Tree-based classification, including random forests
# Support vector machines
# Tuning methodologies including cross-validation and hyperparameter selection

# Introduction 
# See PPT

# regression example
# import modules
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# function to get boston dataset with training and test sets
def get_boston():
    # load the boston dataset
    dataset = load_boston()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['MEDV'] = dataset.target
    df.index.name = 'record'

    # split into training and test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(df.loc[:, df.columns != 'MEDV'], 
                         df['MEDV'], test_size=.33, random_state=42)

    return [X_train, X_test, y_train, y_test]

# Linear Regression #
# import modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# get moon dataset
X_train, X_test, y_train, y_test = get_boston()

#instantiate regression object and fit to training data
clf = LinearRegression()
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
r2 = r2_score(y_test, y_pred) 
print('r2 score is = ' + str(r2))

# regularization with penalized regression
# see PPT and come back to the example below

# Lasso Regression #
# import modules
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# get moon dataset
X_train, X_test, y_train, y_test = get_boston()

#instantiate classifier object and fit to training data
clf = Lasso(alpha=0.3)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
r2 = r2_score(y_test, y_pred) 
print('r2 score is = ' + str(r2))

# Ridge Regression #
# import modules
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# get moon dataset
X_train, X_test, y_train, y_test = get_boston()

#instantiate classifier object and fit to training data
clf = Ridge(alpha=0.3)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
r2 = r2_score(y_test, y_pred) 
print('r2 score is = ' + str(r2))

# classification
# see PPT 

# import modules
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# function to get toy moon set
def get_moon_data():
    # make blobs and split into train and test sets
    X, y = make_moons(n_samples=150, noise=0.4, random_state=42)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.33, random_state=42)

    return [X_train, X_test, y_train, y_test]

# Logistic Regression Classification #
# see PPT
# import modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import seaborn as sns; sns.set()

# get moon dataset
X_train, X_test, y_train, y_test = get_moon_data()

#instantiate classifier object and fit to training data
# clf = LogisticRegression(solver='lbfgs')
# Regularized Logistic Regression #
clf = LogisticRegression(solver='lbfgs', penalty='l2', C=0.5)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred) 
print('f1 score is = ' + str(f1))

# plot confusion matrix #
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Creates a confusion matrix
cm = confusion_matrix(y_pred, y_test) 

# create df and add class names
labels = ['top crescent', 'bottom cresent']
df_cm = pd.DataFrame(cm,
                     index = labels, 
                     columns = labels)

# plot figure
plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, cmap="GnBu", annot=True)

#add titles and labels for the axes
plt.title('Logistic Regression \nF1 Score:{0:.3f}'.format(f1_score(y_test, y_pred)))
plt.ylabel('Prediction')
plt.xlabel('Actual Class')
plt.show()

# Support Vector Machine (SVM) #
# see PPT
# import modules
from sklearn.svm import SVC
from sklearn.metrics import f1_score

# get moon dataset
X_train, X_test, y_train, y_test = get_moon_data()

#instantiate classifier object and fit to training data
clf = SVC(kernel="linear", C=0.5)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred) 
print('f1 score is = ' + str(f1))

# SVM with Gaussian Kernel Classification #
# instantiate classifier object and fit to training data
clf = SVC(gamma=2, C=1)
clf.fit(X_train, y_train)

# predict on test set and score the predictions against y_test
y_pred = clf.predict(X_test)
f1 = f1_score(y_test, y_pred) 
print('f1 score is = ' + str(f1))

# Tree-based classification #
# see PPT
# import modules
# 
