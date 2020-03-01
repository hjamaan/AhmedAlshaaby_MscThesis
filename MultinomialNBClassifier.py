import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB , MultinomialNB

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
model =  MultinomialNB()
model.fit(X, y)
# Print model info
print("********************")
print(model)
# Predictions
expected = y
predicted = model.predict(X)
# Printing the results from metrics
print(metrics.classification_report(expected, predicted))
# Printing the confusion matrix
print(metrics.confusion_matrix(expected, predicted))
print("********************")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


#The Outputs
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
 #             precision    recall  f1-score   support

  #         0       0.91      0.89      0.90       280
   #        1       0.79      0.81      0.80       140

   #micro avg       0.87      0.87      0.87       420
   #macro avg       0.85      0.85      0.85       420
#weighted avg       0.87      0.87      0.87       420

#[[250  30]
 #[ 26 114]]
#********************
#Accuracy: 86.190% (2.076%)