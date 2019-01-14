import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
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
decision_tree_classifier = DecisionTreeClassifier()


                 
parameter_grid  = {'min_samples_split':np.arange(2, 80), 'max_depth': np.arange(2,100), 'criterion':['gini', 'entropy']}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

grid_search = GridSearchCV(decision_tree_classifier, param_grid = parameter_grid, scoring='accuracy', cv = kfold)
grid_search.fit(X, y)


print ("Best Score: {}".format(grid_search.cv_results_))
print("********************")
print ("Best Score: {}".format(grid_search.best_score_))
print("********************")
print ("Best params: {}".format(grid_search.best_params_))
print("********************")
print ("Best Score: {}".format(grid_search.best_estimator_))
expected = y
predicted = grid_search.predict(X)
# Printing the results from metrics
print(metrics.classification_report(expected, predicted))
# Printing the confusion matrix
print(metrics.confusion_matrix(expected, predicted))
print("********************")
'''
#*****************************
Best Score: 0.9952380952380953
********************
Best params: {'criterion': 'entropy', 'max_depth': 4, 'min_samples_split': 14}
********************
Best Score: DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=14,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       280
           1       1.00      0.99      1.00       140

   micro avg       1.00      1.00      1.00       420
   macro avg       1.00      1.00      1.00       420
weighted avg       1.00      1.00      1.00       420

[[280   0]
 [  1 139]]
********************
'''
