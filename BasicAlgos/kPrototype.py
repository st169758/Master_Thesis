from kmodes.kprototypes import KPrototypes
from sklearn.datasets import make_blobs
import pandas as pd

X, targets = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
df = pd.DataFrame(X)
df['C'] = 'abc'

X_ = df.to_numpy()

 # Visualize the data
# plt.scatter(X[:,0], X[:,1], picker=True)
# plt.title('Input data')
# plt.show()

kp = KPrototypes(n_clusters=3, init='Huang', random_state=42)
res = kp.fit_predict(X_, categorical = [2])
print(kp.cost_)


