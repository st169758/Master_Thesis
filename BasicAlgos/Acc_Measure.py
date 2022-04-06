from sklearn.cluster import KMeans, Birch
from Bisecting_K_means.Bis_K_Means_Latest import bisectingKMeans
from sklearn.mixture import GaussianMixture
from functions import precompute_fx
from functions import ft_F1
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")


def get_data():
    # Import training and testing data from Imbalance Data Generator project output CSV file:
    df_train = pd.read_csv('./Training_Data.csv')
    df_test = pd.read_csv('./Testing_Data.csv')

    y_train = df_train['target'].to_numpy()
    X_train = df_train.drop(columns=['index', 'target'])
    X_train = X_train.to_numpy()

    y_test = df_test['target'].to_numpy()
    X_test = df_test.drop(columns=['index', 'target'])
    X_test = X_test.to_numpy()

    return X_train,y_train,X_test,y_test

def get_accuracy(k, df, X_test):

    rf_classifiers = []
    cluster_centers = []
    samples_per_cluster = []
    classes_per_cluster = []
    for clust_id in range(k):
        temp = df[df['y_clust'] == clust_id]
        temp_X = temp.iloc[:, :X_train.shape[1]]
        X_ = temp_X.to_numpy()
        y_ = temp.iloc[:, X_train.shape[1]].to_numpy()
        samples_per_cluster.append(len(X_))
        classes_per_cluster.append(len(np.unique(y_)))
        rf = RF(random_state=42)
        clf = rf.fit(X_, y_)
        rf_classifiers.append(clf)
        X_center = np.mean(X_, axis=0)
        cluster_centers.append(X_center)

    df_test = pd.DataFrame(X_test)
    df_test['y_true'] = y_test
    y_pred = []

    for point in X_test:
        dist = []
        for ind in cluster_centers:
            dist.append(np.linalg.norm(point - ind))
        near_clust_id = np.argmin(dist)
        # print('Nearest Cluster ID: ', near_clust_id)
        res = rf_classifiers[near_clust_id].predict([point])
        y_pred.append(res[0])
    df_test['y_pred'] = y_pred
    acc = accuracy_score(df_test['y_true'], df_test['y_pred'])
    return acc

def KMeans_Acc(k, X_train):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter= 300, random_state=42)
    y_kmeans = kmeans.fit_predict(X_train)

    # For simplicity, create pandas dataframe to store all pred + true values:
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df['y_clust'] = y_kmeans
    acc = get_accuracy(k,df,X_test)
    return acc


def Birch_Acc(k, X_train):
    birch = Birch(n_clusters=k)
    y_birch = birch.fit_predict(X_train)

    # For simplicity, create pandas dataframe to store all pred + true values:
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df['y_clust'] = y_birch
    acc = get_accuracy(k,df,X_test)
    return acc


def GMM_Acc(k, X_train):
    gmm = GaussianMixture(n_components=k, random_state=42)
    y_gmm = gmm.fit_predict(X_train)

    # For simplicity, create pandas dataframe to store all pred + true values:
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df['y_clust'] = y_gmm
    acc = get_accuracy(k,df,X_test)
    return acc

def Bis_KMeans_Acc(X_train, k):
    y_kmeans = bisectingKMeans(X_train, k)

    # For simplicity, create pandas dataframe to store all pred + true values:
    df = pd.DataFrame(X_train)
    df['targets'] = y_train
    df['y_clust'] = y_kmeans
    acc = get_accuracy(k,df,X_test)
    return acc

def Optimize(X_train, y_train):

    def kmeans_acc(k):
        return KMeans_Acc(k=int(k), X_train=X_train)
    def birch_acc(k):
        return Birch_Acc(k=int(k), X_train=X_train)
    def gmm_acc(k):
        return GMM_Acc(k=int(k), X_train=X_train)
    def bis_kmeans_acc(k):
        return Bis_KMeans_Acc(X_train=X_train, k=int(k))

    Cluster_Methods = ['K_Means', 'BIRCH', 'GMM','Bis_K_Means']

    n = np.unique(y_train)
    df_res = pd.DataFrame()
    optimizer = []
    for i in range(len(Cluster_Methods)):
        print(f'Starting with {Cluster_Methods[i]} method')
        print()
        if i == 0:
            func = kmeans_acc
        elif i == 1:
            func = birch_acc
        elif i == 2:
            func = gmm_acc
        elif i == 3:
            func = bis_kmeans_acc

        optimizer.append(BayesianOptimization(
            f= func,
            pbounds={
                "k": (2, len(n) - 1),
            },
            ptypes={'k': int},
            random_state=42,
            verbose=2
        ))
        optimizer[i].maximize(n_iter=10)

        df_res.loc[i,'Cluster_Method'] = Cluster_Methods[i] + '_Acc'
        k_opt = int(optimizer[i].max['params']['k'])
        df_res.loc[i,'Opt-k'] = k_opt

        print(f'Applying clustering with k = {k_opt} on training data..')

        if i == 0:
            kmeans = KMeans(n_clusters=k_opt, init="k-means++", n_init=10, max_iter=300, random_state=42)
            y_clust = kmeans.fit_predict(X_train)
        elif i == 1:
            birch = Birch(n_clusters=k_opt)
            y_clust = birch.fit_predict(X_train)
        elif i == 2:
            gmm = GaussianMixture(n_components=k_opt, random_state=42)
            y_clust = gmm.fit_predict(X_train)
        elif i == 3:
            y_clust = bisectingKMeans(X_train,k_opt)

        print('-------------------------------------------------------------------------------------------------')
        print('Starting to train Random Forest on each cluster formed..')
        df = pd.DataFrame(X_train)
        df['targets'] = y_train
        df['y_clust'] = y_clust
        rf_classifiers = []
        cluster_centers = []
        samples_per_cluster = []
        classes_per_cluster = []

        for clust_id in range(k_opt):
            print(f'Training Random Forest on cluster {clust_id}')
            temp = df[df['y_clust'] == clust_id]
            temp_X = temp.iloc[:, :X_train.shape[1]]
            X_ = temp_X.to_numpy()
            y_ = temp.iloc[:, X_train.shape[1]].to_numpy()
            samples_per_cluster.append(len(X_))
            classes_per_cluster.append(len(np.unique(y_)))
            rf = RF(random_state=42)
            try:
                clf = rf.fit(X_, y_)
                rf_classifiers.append(clf)
                X_center = np.mean(X_, axis=0)
                cluster_centers.append(X_center)
            except ValueError:
                print('Cluster contains empty data and is ignored for further calculations. Moving with next cluster..')
                print()

        print('--------------------------------------------------------------------------------------------------')
        print('Allocating data points in X_test to nearest cluster and predicting the result using classifier trained on it..')
        df_test = pd.DataFrame(X_test)
        df_test['y_true'] = y_test
        y_pred = []

        for point in X_test:
            dist = []
            for ind in cluster_centers:
                dist.append(np.linalg.norm(point - ind))
            near_clust_id = np.argmin(dist)
            # print('Nearest Cluster ID: ', near_clust_id)
            res = rf_classifiers[near_clust_id].predict([point])
            y_pred.append(res[0])
        df_test['y_pred'] = y_pred
        print()
        print('Classifier successfully predicted the test data.')

        print()
        acc = accuracy_score(df_test['y_true'], df_test['y_pred'])
        print('Accuracy of predicted score is: ', accuracy_score(df_test['y_true'], df_test['y_pred']))
        f1 = f1_score(df_test['y_true'], df_test['y_pred'], average='weighted')
        print('F1-score of predicted score is: ', f1_score(df_test['y_true'], df_test['y_pred'], average='weighted'))
        print()

        df_res.loc[i,'Accuracy'] = acc
        df_res.loc[i,'F1_score'] = f1

        df_res.loc[i, 'Avg_samples_cluster'] = np.mean(samples_per_cluster)
        df_res.loc[i, 'Avg_classes_cluster'] = np.mean(classes_per_cluster)

    return df_res

X_train, y_train, X_test, y_test = get_data()
df_res = Optimize(X_train, y_train)
print(df_res)
df_res.to_csv('./Results_Acc.csv', index=False, header=True)


