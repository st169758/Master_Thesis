# This code executes basic operations involved in optimizers by taking use of sample dataset

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart_disease.csv')
print(df.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','sex']], df['target'], test_size=0.10, shuffle=True, random_state=0)
print(X_train.columns)
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train,y_train)

y_pred = dec_tree.predict(X_test)
y_true = y_test

from sklearn.metrics import accuracy_score
print(accuracy_score(y_true, y_pred))

from sklearn.tree import export_graphviz
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

dot_data = export_graphviz(dec_tree, feature_names=['age','sex'],
                           filled=True, rounded=True)

graph = graphviz.Source(dot_data)
graph.render("tree")
print("max depth is: ",dec_tree.max_depth, dec_tree.min_samples_leaf, dec_tree.min_samples_split)

# Since accuracy score is lower, try using grid search thereby running different hyperparameter combinations
from sklearn.model_selection import GridSearchCV
import time
start = time.time()
print("Introducing GridSearchCV..")
# Below we introduce 9*8*8 combinations
hyperparameter_space = {'max_depth':[None, 2,3,4,6,8,10,12,15,20],
                        'min_samples_leaf':[1,2,4,6,8,10,20,30],
                        'min_samples_split':[1,2,3,4,5,6,8,10]}
gs = GridSearchCV(dec_tree, param_grid=hyperparameter_space,scoring="accuracy", n_jobs=-1, cv=10, return_train_score=True)
gs.fit(X_train, y_train)

opt_hyp_setting = gs.best_params_
print("Best hyperparameter settings are: ",opt_hyp_setting, " with accuracy score of: ", gs.best_score_)
end = time.time()
comp_time_grid = end - start
print("and time taken is: ", comp_time_grid)

# Move on with random search for the same set

print("Introducing Random search CV..")

from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(dec_tree,param_distributions=hyperparameter_space, n_iter=10, n_jobs=-1, return_train_score=True)
rs.fit(X_train,y_train)

opt_hyp_setting1 = rs.best_params_

print("Best hyperparameter settings are: ",opt_hyp_setting1, " with accuracy score of: ", rs.best_score_)
end1 = time.time()
comp_time_grid = end1 - end
print("and time taken is: ", comp_time_grid)

# Using random forest for the same dataset
print("Executing Random forest classifier for the same dataset..")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_true = y_test
print("Accuracy of Random forest is: ", accuracy_score(y_true,y_pred))
feature_imp = pd.Series(rf.feature_importances_, index = X_train.columns).sort_values(ascending = False)
print("Feature importance: ", feature_imp)

from bayes_opt import BayesianOptimization
params_gbm ={
    'n_estimators':(100, 200)
}

def black_box_function(n_estimators):
    # n_estimators: one of the RF hyper parameter to optimize for.
    print("n_estimators: ", n_estimators)

    rf = RandomForestClassifier(n_estimators=int(n_estimators))
    rf.fit(X_train, y_train)
    y_score = rf.predict(X_test)
    f = accuracy_score(y_test, y_score)
    return f


bo = BayesianOptimization(f=black_box_function, pbounds=params_gbm)
bo.maximize(init_points = 5, n_iter = 15)

print("Best result: {}; f(x) = {}.".format(bo.max["params"], bo.max["target"]))

# Implementing Hyperband optimizer for hyperparameter tuning (used external dependent library file)
print("Implementing Hyperband optimizer for the same dataset..")

from hyperband import HyperbandSearchCV
rf = RandomForestClassifier()
params = {
    'max_depth': [1,2,3,None],
    'criterion': ['gini','entropy'],
    'bootstrap': [True, False],
    'min_samples_split':[2,4,6,8,10]
}

hyp = HyperbandSearchCV(rf, params, resource_param='n_estimators', scoring='roc_auc', n_jobs=1, verbose=1)
hyp.fit(X_train, y_train)

print('Hyperband best params: ', hyp.best_params_)
print('Hyperband best score: ', hyp.best_score_)

