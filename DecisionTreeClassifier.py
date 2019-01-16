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
from sklearn.preprocessing import Imputer
#loading dataset
df= pd.read_csv('OS1_Data_Class.csv')
y = df.pop('is_data_class').values
#__________________preprocessing dataset________________
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
# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Train the imputor on the df dataset
mean_imputer = mean_imputer.fit(df)
X = mean_imputer.transform(df.values)
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
y = y + 0 
#___________________________________________________
cv_preds= []
cv_scores = []              
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

decision_tree_classifier = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=14,
            min_weight_fraction_leaf=0.0, presort=False, random_state=4,
            splitter='best')

scores = cross_val_score(decision_tree_classifier, X, y, cv=kfold, scoring='accuracy')
print ("Best Score: {}".format(scores))
print("********************")
cv_scores.append(scores.mean()*100)
print("Train accuracy %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std()*100))
print("********************")
# predict on testing set
preds = cross_val_predict(decision_tree_classifier, X, y, cv=kfold)
cv_preds.append(metrics.accuracy_score(y, preds)*100)
print("Test accuracy %0.2f" % (100*metrics.accuracy_score(y, preds)))
print("********************")
print("********************")
#print ("Best Score: {}".format(grid_search.cv_results_))
print("********************")
#print ("Best Score: {}".format(grid_search.best_score_))
print("********************")
#print ("Best params: {}".format(grid_search.best_params_))
print("********************")
#print ("Best Score: {}".format(grid_search.best_estimator_))

# Printing the results from metrics
print("********************Results********")
print(metrics.classification_report(y, preds))
print("********************confusion matrix********")
# Printing the confusion matrix
print(metrics.confusion_matrix(y, preds))
print("********************")

'''
#The outputs 

Best Score: [1.         1.         1.         0.97619048 1.         0.97619048
 1.         1.         1.         1.        ]
********************
Train accuracy 99.52 (+/- 0.95)
********************
Test accuracy 99.52
********************
********************
********************
********************
********************
********************Results********
              precision    recall  f1-score   support

           0       0.99      1.00      1.00       280
           1       1.00      0.99      0.99       140

   micro avg       1.00      1.00      1.00       420
   macro avg       1.00      0.99      0.99       420
weighted avg       1.00      1.00      1.00       420

********************confusion matrix********
[[280   0]
 [  2 138]]
****************
'''
