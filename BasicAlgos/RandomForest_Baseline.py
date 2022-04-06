from sklearn.ensemble import RandomForestClassifier as RF
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Import the data generated using Imbalance Data Generator
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

# Define baseline classifier as Random Forest Classifier trained on the whole dataset
def RandomForestBaseline(X_train, y_train, X_test, y_test):
    rf = RF(random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    df = pd.DataFrame()
    df['y_pred'] = y_pred
    df['y_true'] = y_test
    return df

# Call the pre-defined functions and calculate accuracy + F1-score
X_train,y_train,X_test,y_test = get_data()
df_res = RandomForestBaseline(X_train, y_train, X_test, y_test)
Acc = accuracy_score(df_res['y_true'], df_res['y_pred'])
f1 = f1_score(df_res['y_true'], df_res['y_pred'],average='weighted')
df = pd.DataFrame()
df.loc[0, 'Cluster_Method'] = 'RF_Baseline'
df.loc[0, 'Opt-k'] = '-'
df.loc[0, 'Accuracy'] = Acc
df.loc[0, 'F1_score'] = f1
df.to_csv('./Results.csv', mode='a', index=False, header=False)

print(f'Results are: Accuracy: {Acc}, F1-score: {f1}')

