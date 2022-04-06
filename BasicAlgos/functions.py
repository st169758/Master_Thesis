import matplotlib.pyplot as plt
import numpy as np
import typing as t
import itertools

import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA

# Generate data
X, targets = make_blobs(n_samples = 100, centers = 4, n_features = 2, cluster_std = 1, random_state=42)

# Visualize the data
# plt.scatter(X[:,0], X[:,1], picker=True)
# plt.title('Input data')
# plt.show()

#
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], c = 'purple', label = 'C1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], c = 'orange', label = 'C2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], c = 'green', label = 'C3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], c = 'blue', label = 'C4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], c = 'yellow', label = 'C5')
#
# #Plotting the centroids of the clusters
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], c = 'red', label = 'Centroids')
# plt.legend()
# plt.title('Output: Clustered Data (Optimum scenario)')
# plt.show()


#Start computing Fishers Discriminant Ratio

def precompute_fx(X: np.ndarray, y: np.ndarray) -> t.Dict[str, t.Any]:
    """Precompute some useful things to support complexity measures.
    Parameters
    ----------
    X : :obj:`np.ndarray`, optional
            Attributes from fitted data.
    y : :obj:`np.ndarray`, optional
            Target attribute from fitted data.
    Returns
    -------
    :obj:`dict`
        With following precomputed items:
        - ``ovo_comb`` (:obj:`list`): List of all class OVO combination,
            i.e., [(0,1), (0,2) ...].
        - ``cls_index`` (:obj:`list`):  The list of boolean vectors
            indicating the example of each class.
        - ``cls_n_ex`` (:obj:`np.ndarray`): The number of examples in
            each class. The array indexes represent the classes.
    """

    prepcomp_vals = {}
    classes, class_freqs = np.unique(y, return_counts=True)
    cls_index = [np.equal(y, i) for i in classes]

    # cls_n_ex = np.array([np.sum(aux) for aux in cls_index])
    cls_n_ex = list(class_freqs)
    ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))

    prepcomp_vals["ovo_comb"] = ovo_comb
    prepcomp_vals["cls_index"] = cls_index
    prepcomp_vals["cls_n_ex"] = cls_n_ex

    return prepcomp_vals

def numerator (X: np.ndarray, cls_index, cls_n_ex, i) -> float:
    # print("Num: ", np.sum([cls_n_ex[j]*np.power((np.mean(X[cls_index[j], i]) - np.mean(X[:, i], axis=0)),2) for j in range (len(cls_index))]))
    return np.sum([cls_n_ex[j]*np.power((np.mean(X[cls_index[j], i]) - np.mean(X[:, i], axis=0)),2) for j in range (len(cls_index))])

def denominator (X: np.ndarray, cls_index, cls_n_ex, i) -> float:
    # print("Den: ",np.sum([np.sum(np.power(X[cls_index[j], i]-np.mean(X[cls_index[j], i], axis=0), 2)) for j in range(0, len(cls_n_ex))]))
    return np.sum([np.sum(np.power(X[cls_index[j], i]-np.mean(X[cls_index[j], i], axis=0), 2)) for j in range(0, len(cls_n_ex))])

def compute_rfi (X: np.ndarray, cls_index, cls_n_ex) -> float:
    rfi = []
    for i in range(np.shape(X)[1]):
        num = numerator (X, cls_index, cls_n_ex, i)
        den = denominator(X, cls_index, cls_n_ex, i)
        if den == 0:
            rfi.append(0.0)
        else:
            rfi.append(num / den)

    return rfi

def ft_F1(X: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> float:
    #return 1/(1 + np.max(compute_rfi (X, cls_index, cls_n_ex)))
    # print(compute_rfi (X, cls_index, cls_n_ex))
    if len(X) > 1:
        return np.max(compute_rfi (X, cls_index, cls_n_ex))
    else:
        return 0.0


def _minmax(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the minimum of the maximum values per class
    for all features.
    """
    max_cls = np.zeros((2, X.shape[1]))
    max_cls[0, :] = np.max(X[class1], axis=0)
    max_cls[1, :] = np.max(X[class2], axis=0)
    aux = np.min(max_cls, axis=0)

    return aux


def _minmin(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the minimum of the minimum values per class
    for all features.
    """
    min_cls = np.zeros((2, X.shape[1]))
    min_cls[0, :] = np.min(X[class1], axis=0)
    min_cls[1, :] = np.min(X[class2], axis=0)
    aux = np.min(min_cls, axis=0)

    return aux


def _maxmin(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the maximum of the minimum values per class
    for all features.
    """
    min_cls = np.zeros((2, X.shape[1]))
    min_cls[0, :] = np.min(X[class1], axis=0)
    min_cls[1, :] = np.min(X[class2], axis=0)
    aux = np.max(min_cls, axis=0)

    return aux

def _maxmax(X: np.ndarray, class1: np.ndarray, class2: np.ndarray) -> np.ndarray:
    """ This function computes the maximum of the maximum values per class
    for all features.
    """
    max_cls = np.zeros((2, X.shape[1]))
    max_cls[0, :] = np.max(X[class1], axis=0)
    max_cls[1, :] = np.max(X[class2], axis=0)
    aux = np.max(max_cls, axis=0)
    return aux


def ft_F2(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray) -> float:
    f2_list = []
    if len(ovo_comb) !=0:
        for idx1, idx2 in ovo_comb:
            y_class1 = cls_index[idx1]
            y_class2 = cls_index[idx2]
            zero_ = np.zeros(np.shape(X)[1])
            overlap_ = np.maximum(zero_, _minmax(X, y_class1, y_class2) - _maxmin(X, y_class1, y_class2))
            range_ = _maxmax(X, y_class1, y_class2) - _minmin(X, y_class1, y_class2)
            ratio = overlap_ / range_
            f2_list.append(np.prod(ratio))

        return np.mean(f2_list)
    else:
        return 0.0




def _compute_f3(X_: np.ndarray, minmax_: np.ndarray, maxmin_: np.ndarray) -> np.ndarray:
    """ This function computes the F3 complexity measure given minmax and maxmin."""

    overlapped_region_by_feature = np.logical_and(X_ >= maxmin_, X_ <= minmax_)
    n_fi = np.sum(overlapped_region_by_feature, axis=0)
    idx_min = np.argmin(n_fi)

    return idx_min, n_fi, overlapped_region_by_feature


def ft_F3(X: np.ndarray, ovo_comb: np.ndarray, cls_index: np.ndarray, cls_n_ex: np.ndarray) -> np.ndarray:
    if len(ovo_comb) != 0:
        f3 = []
        for idx1, idx2 in ovo_comb:
            idx_min, n_fi, _ = _compute_f3(X, _minmax(X, cls_index[idx1], cls_index[idx2]),
                                           _maxmin(X, cls_index[idx1], cls_index[idx2]))
            f3.append(n_fi[idx_min] / (cls_n_ex[idx1] + cls_n_ex[idx2]))

        return np.mean(f3)
    else:
        return 0.0


def ft_N1(X: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    # 0-1 scaler
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    # X_ = scaler.transform(X)

    # compute the distance matrix and the minimum spanning tree.
    dist_m = np.triu(distance.cdist(X, X, metric), k=1)
    mst = minimum_spanning_tree(dist_m)
    node_i, node_j = np.where(mst.toarray() > 0)

    # which edges have nodes with different class
    which_have_diff_cls = y[node_i] != y[node_j]

    # number of different vertices connected
    aux = np.unique(np.concatenate([node_i[which_have_diff_cls], node_j[which_have_diff_cls]])).shape[0]
    if X.shape[0] == 0:
        return 0.0
    else:
        return aux / X.shape[0]


def nearest_enemy(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray,
                  i: int, metric: str = "euclidean", n_neighbors=1):
    " This function computes the distance from a point x_i to their nearest enemy"
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    # X = scaler.transform(X)

    label_query = y[i]
    df_que = pd.DataFrame(X)
    df_que['y_true'] = y
    df_que = df_que[df_que['y_true'] != label_query]
    X_ = df_que.iloc[:, :-1].to_numpy()
    y_ = df_que.iloc[:, -1].to_numpy()

    # X_ = X[np.logical_not(cls_index[y[i]])]
    # y_ = y[np.logical_not(cls_index[y[i]])]
    # For cluster with one only class, no neighbors of other class are present and hence distance and neighbor position is set to 0
    if len(X_) == 0:
        dist_enemy = np.reshape(0, (n_neighbors,))
        pos_enemy = np.reshape(0, (n_neighbors,))
    else:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        neigh.fit(X_, y_)
        dist_enemy, pos_enemy = neigh.kneighbors([X[i, :]])
        dist_enemy = np.reshape(dist_enemy, (n_neighbors,))
        pos_enemy_ = np.reshape(pos_enemy, (n_neighbors,))
        query = X_[pos_enemy_, :]
        pos_enemy = np.where(np.all(X == query, axis=1))
        pos_enemy = np.reshape(pos_enemy, (n_neighbors,))
    return dist_enemy, pos_enemy


def nearest_neighboor_same_class(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray,
                                 i: int, metric: str = "euclidean", n_neighbors=1):
    " This function computes the distance from a point x_i to their nearest neighboor from its own class"
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    # X = scaler.transform(X)

    query = X[i, :]
    label_query = y[i]
    df_que = pd.DataFrame(X)
    df_que['y_true'] = y
    df_que = df_que[df_que['y_true']==label_query]

    X_ = df_que.iloc[:,:-1].to_numpy()
    y_ = df_que.iloc[:,-1].to_numpy()

    # X_ = X[cls_index[label_query]]
    # y_ = y[cls_index[label_query]]

    pos_query = np.where(np.all(X_ == query, axis=1))
    X_ = np.delete(X_, pos_query, axis=0)
    y_ = np.delete(y_, pos_query, axis=0)

    # For class with one instance, no neighbors are present and hence distance and neighbor position is set to 0
    if len(X_) == 0:
        dist_neigh = np.reshape(0, (n_neighbors,))
        pos_neigh = np.reshape(0, (n_neighbors,))
    else:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        neigh.fit(X_, y_)
        dist_neigh, pos_neigh = neigh.kneighbors([X[i, :]])
        dist_neigh = np.reshape(dist_neigh, (n_neighbors,))
        pos_neigh = np.reshape(pos_neigh, (n_neighbors,))
    return dist_neigh, pos_neigh

def intra_extra(X: np.ndarray, y: np.ndarray, cls_index: np.ndarray):
    intra = np.sum([nearest_neighboor_same_class (X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    extra = np.sum([nearest_enemy (X, y, cls_index, i)[0] for i in range(np.shape(X)[0])])
    return intra/(1+extra)

def ft_N2 (X: np.ndarray, y: np.ndarray, cls_index: np.ndarray):
    intra_extra_ = intra_extra(X, y, cls_index)
    return intra_extra_


def ft_N3(X: np.ndarray, y: np.ndarray, metric: str = "euclidean") -> float:
    loo = LeaveOneOut()
    loo.get_n_splits(X, y)
    y_test_ = []
    pred_y_ = []

    if loo.get_n_splits(X, y) < 2:
        error = 0
    else:
        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = KNeighborsClassifier(n_neighbors=1, metric=metric)
            model.fit(X_train, y_train)
            pred_y = model.predict(X_test)
            y_test_.append(y_test)
            pred_y_.append(pred_y)

        error = 1 - accuracy_score(y_test_, pred_y_)
    return error

def precompute_pca_tx(X: np.ndarray) -> t.Dict[str, t.Any]:
    """Precompute PCA to Tx complexity measures.
    Parameters
    ----------
    X : :obj:`np.ndarray`, optional
            Attributes from fitted data.
    Returns
    -------
    :obj:`dict`
        With following precomputed items:
        - ``m`` (:obj:`int`): Number of features.
        - ``m_`` (:obj:`int`):  Number of features after PCA with 0.95.
        - ``n`` (:obj:`int`): Number of examples.
    """
    prepcomp_vals = {}

    pca = PCA(n_components=0.95)
    pca.fit(X)

    m_ = pca.explained_variance_ratio_.shape[0]
    m = X.shape[1]
    n = X.shape[0]

    prepcomp_vals["m_"] = m_
    prepcomp_vals["m"] = m
    prepcomp_vals["n"] = n

    return prepcomp_vals

def ft_T2(m: int, n: int) -> float:
    return m/n

def ft_T3(m_: int, n: int) -> float:
    return m_/n

def ft_T4(m: int, m_: int) -> float:
    return m_/m

if __name__=="__main__":

    print('Compute FDR for different K values')
    res = []
    for k in range(2,16):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter= 300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        precomp_fx = precompute_fx(X, y_kmeans)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']
        F1 = ft_F1(X, cls_index, cls_n_ex)
        print(F1)
        res.append(1/(1+F1))

    min_k = np.argmin(res) + 2
    print('Best K within given range is achieved for: k = ',min_k)

    print()
    print('---------------------------------------------------------------')
    print()
    print('Moving with computation of Volume of Overlapping Region (F2):')

    print('Compute F2 for different K values')
    res_F2 = []
    for k in range(2,10):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter= 300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        precomp_fx = precompute_fx(X, y_kmeans)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']
        F2 = ft_F2(X, ovo_comb, cls_index)
        print(F2)
        res_F2.append(F2)

    min_k = (len(res_F2) - res_F2[::-1].index(min(res_F2)) - 1) + 2
    print('Best K within given range is achieved for: k = ',min_k)

    print()
    print('---------------------------------------------------------------')
    print()
    print('Moving with computation of Maximum Individual Feature Efficiency (F3):')

    print('Compute F3 for different K values')
    res_F3 = []
    for k in range(2,10):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter= 300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        precomp_fx = precompute_fx(X, y_kmeans)
        cls_index = precomp_fx['cls_index']
        cls_n_ex = precomp_fx['cls_n_ex']
        ovo_comb = precomp_fx['ovo_comb']
        F3 = ft_F3(X, ovo_comb, cls_index, cls_n_ex)
        print(F3)
        res_F3.append(F3)

    min_k = (len(res_F3) - res_F3[::-1].index(min(res_F3)) - 1) + 2
    print('Best K within given range is achieved for: k = ',min_k)

    print()
    print('---------------------------------------------------------------')
    print()

    print('Computing N1 for different K values:')

    res_N1 = []
    for k in range(2, 16):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        N1 = ft_N1(X, y_kmeans)
        print(N1)
        res_N1.append(N1)

    min_k = (len(res_N1) - res_N1[::-1].index(min(res_N1)) - 1) + 2
    print('Best K within given range is achieved for: k = ', min_k)

    print()
    print('---------------------------------------------------------------')
    print()

    print('Computing N2 for different K values:')

    res_N2 = []
    for k in range(2, 12):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        precomp_fx = precompute_fx(X, y_kmeans)
        cls_index = precomp_fx['cls_index']
        N2 = ft_N2(X, y_kmeans, cls_index)
        print(N2)
        res_N2.append(N2)

    min_k = (len(res_N2) - res_N2[::-1].index(min(res_N2)) - 1) + 2
    print('Best K within given range is achieved for: k = ', min_k)

    print()
    print('---------------------------------------------------------------')
    print()

    print('Computing N3 for different K values:')

    res_N3 = []
    for k in range(2, 12):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        N3 = ft_N3(X, y_kmeans)
        print(N3)
        res_N3.append(N3)

    min_k = (len(res_N3) - res_N3[::-1].index(min(res_N3)) - 1) + 2
    print('Best K within given range is achieved for: k = ', min_k)

    print()
    print('---------------------------------------------------------------')
    print()

    print('Computing T2 for different K values:')
    # For topology based metrics, we calculate them once the clusters are formed.
    # Then take mean of all the clusters for one k combination.

    res_T2 = []
    res_T3 = []
    res_T4 = []
    for k in range(2, 6):
        print()
        print('k = ', k)
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        y_kmeans = kmeans.fit_predict(X)

        # Consider one cluster at a time, find T2, T3 and T4, then average over all clusters.
        t2_temp = []
        t3_temp = []
        t4_temp = []

        for clus_id in range(k):
            X_ = X[y_kmeans==clus_id]
            precomp_pca = precompute_pca_tx(X_)
            m = precomp_pca['m']
            n = precomp_pca['n']
            m_ = precomp_pca['m_']
            T2 = ft_T2(m, n)
            T3 = ft_T3(m_, n)
            T4 = ft_T4(m, m_)

            t2_temp.append(T2)
            t3_temp.append(T3)
            t4_temp.append(T4)

        res_T2.append(np.mean(t2_temp))
        res_T3.append(np.mean(t3_temp))
        res_T4.append(np.mean(t4_temp))

    print(res_T2)
    print(res_T3)
    print(res_T4)

    min_k_t2 = (len(res_T2) - res_T2[::-1].index(max(res_T2)) - 1) + 2
    print('Best K within given range is achieved for T2: k = ', min_k_t2)
    min_k_t3 = (len(res_T3) - res_T3[::-1].index(max(res_T3)) - 1) + 2
    print('Best K within given range is achieved for T3: k = ', min_k_t3)
    min_k_t4 = (len(res_T4) - res_T4[::-1].index(min(res_T4)) - 1) + 2
    print('Best K within given range is achieved for T3: k = ', min_k_t4)


    print()
    print('---------------------------------------------------------------')
    print()