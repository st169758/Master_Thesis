from sklearn.metrics import adjusted_rand_score
import pandas as pd

# Import the training dataset of imbalance data generator containing group information (integer mapped values)
df_IDG = pd.read_csv('./Training_data_ARI.csv')
group_idg = list(df_IDG['group1'])

# Import data from kmeans_bo_f1 clustered data labels
df_kmeans_f1 = pd.read_csv('./Training_Data_ARI_KMeans.csv')
clust_lab = list(df_kmeans_f1['Cluster_ID'])

# Calculate Adjusted Rand Index score of both
res = adjusted_rand_score(group_idg, clust_lab)
print(res)



