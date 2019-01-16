#-------------------------------------
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import Imputer
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
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
#--------------load datast---------------------------
df= pd.read_csv('OS1_Data_Class.csv')
y = df.pop('is_data_class').values
df.pop('IDType')
df.pop('project')
df.pop('package')
df.pop('complextype')
#----------------------- missing values-------------
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

#------------bulid clasifier------
cv_preds= []
cv_scores = [] 
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
# build a classifier
rfc=RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
            max_depth=8, max_features=16, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=None,
            oob_score=False, random_state=1, verbose=0,
            warm_start=False)
scores = cross_val_score(rfc, X, y, cv=kfold, scoring='accuracy')

#-------------------obtaining the Results----- 
cv_scores.append(scores.mean()*100)
print ("Best Score: {}".format(scores))
print("********************")
print("Train accuracy %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))
print("********************")
# predict on testing set
preds = cross_val_predict(rfc, X, y, cv=kfold)
cv_preds.append(metrics.accuracy_score(y, preds)*100)
print("Test accuracy %0.2f" % (100*metrics.accuracy_score(y, preds)))
print("********************")
print("********************")
print("********************")
# Printing the results from metrics
print("********************Results********")
print(metrics.classification_report(y, preds))
print("********************confusion matrix********")
# Printing the confusion matrix
print(metrics.confusion_matrix(y, preds))
print("********************")
'''--- the outputs --------
Best Score: [1.         1.         1.         1.         0.97619048 1.
 1.         1.         1.         1.        ]
********************
Train accuracy 99.76 (+/- 0.71)
********************
Test accuracy 99.76
********************
********************
********************
********************Results********
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       280
           1       0.99      1.00      1.00       140

   micro avg       1.00      1.00      1.00       420
   macro avg       1.00      1.00      1.00       420
weighted avg       1.00      1.00      1.00       420

********************confusion matrix********
[[279   1]
 [  0 140]]
********************
'''