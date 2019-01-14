import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
df= pd.read_csv('OS1_Data_Class.csv')
y = df.pop('is_data_class').values
df.pop('IDType')
df.pop('project')
df.pop('package')
df.pop('complextype')
X = np.array(df)
#missing values
X[X == '?'] = -1
X = X.astype('float')
#Rescaling data
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
#-------------------
#conert lables to 0 or 1
y = y + 0 
#y = np.expand_dims(y, 1)
#-----------------------
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
