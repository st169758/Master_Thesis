# This code executes basic clustering algorithm

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Configuration options
num_samples_total = 1000
cluster_centers = [(20,20), (4,4)]
num_features = len(cluster_centers)
num_classes = len(cluster_centers)


# Generate data
X, targets = make_blobs(n_samples = num_samples_total, centers = cluster_centers, n_features = num_features, center_box=(0, 1), cluster_std = 2)

# Visualize the data
plt.scatter(X[:,0], X[:,1], marker="x", picker=True)
plt.title('Input data')
plt.show()

np.save('clusters.npy', X)
X = np.load('clusters.npy')

# Fit K-means with Scikit
kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
kmeans.fit(X)

# Predict the cluster for all the samples
P = kmeans.predict(X)

# Generate scatter plot for training data
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', P))
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.title('Two clusters of data')
plt.xlabel('Temperature yesterday')
plt.ylabel('Temperature today')
plt.show()