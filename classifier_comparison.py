import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV,classification_report,StratifiedKFold,cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
datasets = ['OS1_Data_Class.csv','OS1_God_Class.csv', 'OS1_Long_Method.csv', 'OS1_Feature_Envy.csv','OS2_ArgoUML_Functional_Decomposition.csv', 'OS2_ArgoUML_God_Class.csv', 'OS2_ArgoUML_Spaghetti_Code.csv', 'OS2_ArgoUML_Swiss_Army_Knife.csv'
        ,'OS2_Azureus_Functional_Decomposition.csv','OS2_Azureus_God_Class.csv','OS2_Azureus_Spaghetti_Code.csv','OS2_Azureus_Swiss_Army_Knife.csv','OS2_Xerces_Functional_Decomposition.csv',
        'OS2_Xerces_God_Class.csv','OS2_Xerces_Spaghetti_Code.csv','OS2_Xerces_Swiss_Army_Knife.csv']

# prepare models
#--------------------------------------------
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('MLP', MLPClassifier()))
models.append(('SGDC', SGDClassifier()))
models.append(('KNNN', KNeighborsClassifier()))
models.append(('SVM_linear', SVC()))
models.append(('SVM_GAMA', SVC()))
models.append(('GaussianP', GaussianProcessClassifier()))
models.append(('CART', DecisionTreeClassifier(max_depth=5)))
models.append(('RF', RandomForestClassifier()))
models.append(('MLPP', MLPClassifier(alpha=1)))
models.append(('ADB', AdaBoostClassifier()))
models.append(('Quadra', QuadraticDiscriminantAnalysis()))
#---------------------------

# Specify the N fold
num_folds = 10
seed = 7
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
#kfold = cross_val_score.KFold( n_folds=num_folds, random_state=seed)
df_list = []

# Specify the N fold
num_folds = 10
seed = 7
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
#kfold = cross_val_score.KFold( n_folds=num_folds, random_state=seed)
for name, model in models:
	names.append(name)
 



