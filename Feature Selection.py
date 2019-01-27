# By ALshaaby
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import warnings
from sklearn.feature_selection import SelectPercentile, f_classif,chi2, mutual_info_classif
from sklearn import  svm

datasets = ['OS1_Data_Class.csv','OS1_God_Class.csv', 'OS1_Long_Method.csv', 'OS1_Feature_Envy.csv','OS2_ArgoUML_Functional_Decomposition.csv', 'OS2_ArgoUML_God_Class.csv', 'OS2_ArgoUML_Spaghetti_Code.csv', 'OS2_ArgoUML_Swiss_Army_Knife.csv'
        ,'OS2_Azureus_Functional_Decomposition.csv','OS2_Azureus_God_Class.csv','OS2_Azureus_Spaghetti_Code.csv','OS2_Azureus_Swiss_Army_Knife.csv','OS2_Xerces_Functional_Decomposition.csv',
        'OS2_Xerces_God_Class.csv','OS2_Xerces_Spaghetti_Code.csv','OS2_Xerces_Swiss_Army_Knife.csv']

def preprocessing_(dff):
    
    dff=dff.replace('?', np.nan)
    dff=dff.replace("?", np.nan)
    dff=dff.replace(" ", np.nan)
    dff=dff.replace("", np.nan)
    dff=dff.replace('', np.nan)
    dff=dff.replace(' ', np.nan)
# Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Train the imputor on the df dataset
    mean_imputer = mean_imputer.fit(dff)
    x_ = mean_imputer.transform(dff.values)
# scalling 
    scaler=MinMaxScaler(feature_range=(0,1))
    x_=scaler.fit_transform(x_)
    return x_
def remove_unwanted_attributes_class(dff):
    
    dff.pop('IDType')
    dff.pop('project')
    dff.pop('package')
    dff.pop('complextype')
    return dff
  
df= pd.read_csv('OS1_Data_Class.csv')
y = df.pop('is_data_class').values    
df=remove_unwanted_attributes_class(df)
X=preprocessing_(df)
y = y + 0 
warnings.simplefilter(action='ignore', category=FutureWarning)
