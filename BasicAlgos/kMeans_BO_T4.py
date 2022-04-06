import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_classification
from sklearn.cluster import KMeans
from functions import precompute_pca_tx
from functions import ft_T4
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def get_data(n_samples, n_classes, n_features):
    # Generate data
    # X, targets = make_blobs(n_samples = n_samples, centers = n_classes, n_features =n_features, cluster_std = 1.5, random_state=42)
    X, targets = make_classification(n_samples=1000, n_features=100, n_informative=40, n_repeated=40, n_redundant=5, n_classes=84,n_clusters_per_class=1, random_state=42)

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X, targets

def KMeans_T4(k, X_train):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter= 300, random_state=42)
    y_kmeans = kmeans.fit_predict(X_train)

    # For simplicity, create pandas dataframe to store all pred + true values:
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df['y_kmeans'] = y_kmeans
    df_clust = []
    for clust_id in range(k):
        temp = df[df['y_kmeans']==clust_id]
        temp_X = temp.iloc[:,:X_train.shape[1]]
        X_ = temp_X.to_numpy()
        y_ = temp.iloc[:,X_train.shape[1]].to_numpy()
        # print(f'clust_id: {clust_id} for k = {k}; list has {len(X_)} values')
        precomp_tx = precompute_pca_tx(X_)
        m = precomp_tx['m']
        m_ = precomp_tx['m_']

        F4 = ft_T4(m, m_)
        df_clust.append(F4)

    return 1/(1+np.mean(df_clust))

def Optimize_kmeans(X_train, y_train):
    #Apply Bayesian Optimization to K-Means
    def kmeans_t4(k):
        return KMeans_T4(k=int(k), X_train=X_train)
    n = np.unique(y_train)
    optimizer = BayesianOptimization(
        f=kmeans_t4,
        pbounds={
            "k": (2, len(n)-1),
        },
        random_state=42,
        verbose=2
    )
    optimizer.maximize(n_iter=20)

    # Store results in an CSV file.
    res = optimizer.res
    df_res = pd.DataFrame(columns=['k','target'])
    df_res['k'] = [res[i]['params']['k'] for i in range(len(res))]
    df_res['target'] = [res[i]['target'] for i in range(len(res))]
    df_res.to_csv('./T4_results.csv',header=True)

    return int(optimizer.max['params']['k'])

# Call predefined functions and get the results using Bayesian Optimization:

X_train, X_test, y_train, y_test, X, targets = get_data(n_samples=1000, n_classes=84, n_features=100)
k_opt = Optimize_kmeans(X_train, y_train)
# k_opt = 2
print()
print('Optimum k value is: k = ', k_opt)

print('-------------------------------------------------------------------------------------------------')
# Cluster as per the optimum k-value:
print(f'Applying clustering with k = {k_opt} on training data..')
kmeans = KMeans(n_clusters=k_opt, init="k-means++", n_init=10, max_iter= 300, random_state=42)
y_kmeans = kmeans.fit_predict(X_train)
print()
print(f'{k_opt} clusters successfully created')

print('-------------------------------------------------------------------------------------------------')
print('Starting to train Random Forest on each cluster formed..')

# Train Random Forest on each cluster formed:
# For simplicity, create pandas dataframe to store all pred + true values:
# Store all RF classifiers into list and also store all cluster centers into separate list.

df = pd.DataFrame(X_train)
df['targets'] = y_train
df['y_kmeans'] = y_kmeans
rf_classifiers = []
cluster_centers = []
for clust_id in range(k_opt):
    print(f'Training Random Forest on cluster {clust_id}')
    temp = df[df['y_kmeans']==clust_id]
    temp_X = temp.iloc[:,:X_train.shape[1]]
    X_ = temp_X.to_numpy()
    y_ = temp.iloc[:,X_train.shape[1]].to_numpy()
    rf = RF(random_state=42)
    clf = rf.fit(X_, y_)
    rf_classifiers.append(clf)
    X_center = np.mean(X_, axis=0)
    cluster_centers.append(X_center)

# Consider 'X_test' data and allocate each data point to nearest cluster center, store predictions and targets in dataframe:
print('--------------------------------------------------------------------------------------------------')
print('Allocating data points in X_test to nearest cluster and predicting the result using classifier trained on it..')
df_test = pd.DataFrame(X_test)
df_test['y_true'] = y_test
y_pred = []
for point in X_test:
    dist = []
    for i in cluster_centers:
        dist.append(np.linalg.norm(point - i))
    near_clust_id = np.argmin(dist)
    # print('Nearest Cluster ID: ', near_clust_id)
    res = rf_classifiers[near_clust_id].predict([point])
    y_pred.append(res[0])
df_test['y_pred'] = y_pred
print()
print('Classifier successfully predicted the test data. Showing the overall result in a dataframe:')

# 'y_true' is the true target and 'y_pred' is the predicted one using classifier trained on each cluster.

print('--------------------------------------------------------------------------------------------------')
print(df_test.head(20))
print()
print('Accuracy of predicted score is: ',accuracy_score(df_test['y_true'],df_test['y_pred']))
