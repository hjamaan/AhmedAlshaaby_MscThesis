import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy.stats import randint as sp_randint

df= pd.read_csv('OS1_Data_Class.csv')
y = df.pop('is_data_class').values
df.pop('IDType')
df.pop('project')
df.pop('package')
df.pop('complextype')
df=df.replace('?', np.nan)
df=df.replace("?", np.nan)
df=df.replace(" ", np.nan)
df=df.replace("", np.nan)
df=df.replace('', np.nan)
df=df.replace(' ', np.nan)
y = y + 0 
# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(df)
X = mean_imputer.transform(df.values)

clf=RandomForestClassifier()
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# use a full grid over all parameters
param_grid = {'max_depth': np.arange(2,20),
              "max_features": [1, 3, 10,20],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
start = time()
grid_search.fit(X, y)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

print ("cv result: {}".format(grid_search.cv_results_))
print("********************")
print ("Best Score: {}".format(grid_search.best_score_))
print("********************")
print ("Best params: {}".format(grid_search.best_params_))
print("********************")
print ("Best estimator: {}".format(grid_search.best_estimator_))
expected = y
predicted = grid_search.predict(X)
# Printing the results from metrics
print(metrics.classification_report(expected, predicted))
# Printing the confusion matrix
print(metrics.confusion_matrix(expected, predicted))
print("********************")
