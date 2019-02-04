# By ALshaaby
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
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score, cross_val_predict
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.feature_selection import SelectPercentile, f_classif,chi2, mutual_info_classif,SelectKBest,GenericUnivariate
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn import  svm

#Defin the list of datasets 
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
def remove_unwanted_attributes_method(dff):
    
    dff.pop('IDMethod')
    dff.pop('project')
    dff.pop('package')
    dff.pop('complextype')
    dff.pop('method')
    return dff 
  
df= pd.read_csv('OS1_Data_Class.csv')
y = df.pop('is_data_class').values    
df=remove_unwanted_attributes_class(df)
X=preprocessing_(df)
y = y + 0 
warnings.simplefilter(action='ignore', category=FutureWarning)

models = []
models.append(('RFE', RFE(RandomForestClassifier(), 20)))
models.append(('SelectKBest',  SelectKBest(score_func=chi2, k=3)))
models.append(('SelectPercentile', SelectPercentile(f_classif, percentile=10)))
models.append(('GenericUnivariateSelect', GenericUnivariateSelect(chi2, 'k_best', param=20)))
models.append(('PCA',  PCA(n_components=2)))
models.append(('Feature Importance', ExtraTreesClassifier()))

for dataset in datasets:

#---------------Impot the dataset-------------   
  df= pd.read_csv(dataset)
  
  # ------Select the dataset------------
  
  if dataset== 'OS1_Data_Class.csv':
    y = df.pop('is_data_class').values    
    df=remove_unwanted_attributes_class(df)
    X=preprocessing_(df)
    y = y + 0 

  elif dataset== 'OS1_God_Class.csv':
    y = df.pop('is_god_class').values
    df=remove_unwanted_attributes_class(df)
    X=preprocessing_(df)
    y = y + 0 
    
  elif dataset== 'OS1_Feature_Envy.csv':
    
    y = df.pop('is_feature_envy').values
    df=remove_unwanted_attributes_method(df)
    X=preprocessing_(df)
    y = y + 0 
    
  elif dataset== 'OS1_Long_Method.csv':
    y = df.pop('is_long_method').values
    df=remove_unwanted_attributes_method(df)
    X=preprocessing_(df)
    y = y + 0 
    
  elif dataset== 'OS2_ArgoUML_Functional_Decomposition.csv':
    y = df.pop('FD').values  
    X=preprocessing_(df)
  
  elif dataset== 'OS2_ArgoUML_God_Class.csv':
    y = df.pop('BLOB').values  
    X=preprocessing_(df)
 
  elif dataset== 'OS2_ArgoUML_Spaghetti_Code.csv':
    y = df.pop('SC').values  
    X=preprocessing_(df)
  
  elif dataset== 'OS2_ArgoUML_Swiss_Army_Knife.csv':
    y = df.pop('SAK').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Azureus_Functional_Decomposition.csv':
    y = df.pop('FD').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Azureus_God_Class.csv':
    y = df.pop('BLOB').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Azureus_Spaghetti_Code.csv':
    y = df.pop('SC').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Azureus_Swiss_Army_Knife.csv':
    y = df.pop('SAK').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Xerces_Functional_Decomposition.csv':
    y = df.pop('FD').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Xerces_God_Class.csv':
    y = df.pop('BLOB').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Xerces_Spaghetti_Code.csv':
    y = df.pop('SC').values  
    X=preprocessing_(df)
    
  elif dataset== 'OS2_Xerces_Swiss_Army_Knife.csv':
    y = df.pop('SAK').values  
    X=preprocessing_(df)
#-----------------End selecting dataset---------------------------------
  print('----------------------------------------')
  print('----------------------------------------')
  print('----------------------------------------')
  print(dataset)
  print('----------------------------------------')
  print('----------------------------------------')
  print('----------------------------------------')


  
  for name, model in models:
    if name== 'RFE':
        print('----------------------------------------')
        print(name)
        print('----------------------------------------')
        print('----------------------------------------')
        rfe = model.fit(X, y)
        # summarize the selection of the attributes
        print("Num Features: %d" % (rfe.n_features_))
        print("Selected Features: %s" % (rfe.support_))
        print("Feature Ranking: %s" % (rfe.ranking_))

  
    elif name== 'SelectPercentile':
        print('----------------------------------------')
        print(name)
        print('----------------------------------------')
        print('----------------------------------------')
        fit = model.fit(X, y)
        # summarize scores
        np.set_printoptions(precision=20)
        print(fit.scores_)
        print(fit.pvalues_)
        features = fit.transform(X)
        # summarize selected features
        print(features[0:5,:])
        
    elif name== 'SelectKBest':
        print('----------------------------------------')
        print(name)
        print('----------------------------------------')
        print('----------------------------------------')
        fit = model.fit(X, y)
        # summarize scores
        np.set_printoptions(precision=20)
        print(fit.scores_)
        print(fit.pvalues_)
        features = fit.transform(X)
        # summarize selected features
        print(features[0:5,:])

        

        
    elif name== 'GenericUnivariateSelect':
        print('----------------------------------------')
        print(name)
        print('----------------------------------------')
        print('----------------------------------------')
        fit = model.fit(X, y)
        # summarize scores
        np.set_printoptions(precision=3)
        print(fit.scores_)
        print(fit.pvalues_)
        features = fit.transform(X)
        # summarize selected features
        print(features[0:5,:]) 
       
        
    elif name== 'PCA':
        print('----------------------------------------')
        print(name)
        print('----------------------------------------')
        print('----------------------------------------')
        fit = model.fit(X)
        # summarize components
        print("Explained Variance: %s"% (fit.explained_variance_ratio_))
        
        print("Explained Variance: %s"%(fit.components_))
        

    elif name== 'Feature Importance':
        print('----------------------------------------')
        print(name)
        print('----------------------------------------')
        print('----------------------------------------')
        model.fit(X, y)
        print(model.feature_importances_)
        
        
        
        

