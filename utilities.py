from sklearn.model_selection import cross_val_score
from scipy.stats.stats import pearsonr
from sklearn.metrics import mutual_info_score
import pandas as pd
from scipy.sparse import issparse
import numpy as np
import pylab as pl
from sklearn import metrics, model_selection
from scipy.io import arff
from sklearn.metrics import recall_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
# ENSEMBLE CLASS
class ensemble_clfs:
    def __init__(self, clf_list):
        self.clf_list = clf_list
        self.n_clfs = len(clf_list)
        self.trained_clfs = [None] * self.n_clfs
        self.trained_ids = [] 
       

    def fit(self, X, y, clf_id):
        clf = self.clf_list[clf_id]
        clf.fit(X, y)
        self.trained_clfs[clf_id] = clf
        self.trained_ids += [clf_id]

    def predict(self, X):
        n_trained = len(self.trained_clfs)
        pred_list = np.zeros((X.shape[0], n_trained)) 

        for i in self.trained_ids:
            clf = self.trained_clfs[i]
              ##### compute the probablity
            y_pred = clf.predict_proba(X)[:, 1]
            pred_list[:, i] = y_pred

        return np.mean(pred_list, axis=1)

##### READING DATASET
def preprocessing_(dff):

    dff = dff.replace('?', np.nan)
    dff = dff.replace("?", np.nan)
    dff = dff.replace(" ", np.nan)
    dff = dff.replace("", np.nan)
    dff = dff.replace('', np.nan)
    dff = dff.replace(' ', np.nan)
    # Create an imputer object that looks for 'Nan' values, then replaces them with the mean value of the feature by columns (axis=0)
    mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # Train the imputor on the df dataset
    mean_imputer = mean_imputer.fit(dff)
    x_ = mean_imputer.transform(dff.values)
    # scalling
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_ = scaler.fit_transform(x_)
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
def read_dataset(directory, dataset_name):

    if dataset_name in ['OS1_Data_Class','OS1_Data_Class2']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('is_data_class').values

        df = remove_unwanted_attributes_class(df)
        X = preprocessing_(df)
        y = y + 0
        print(X.shape)

    elif dataset_name in ['OS1_God_Class','OS1_God_Class2','OS1_God_Class3']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('is_god_class').values
        if dataset_name in ['OS1_God_Class', 'OS1_God_Class2']:
           df = remove_unwanted_attributes_class(df)

        X = preprocessing_(df)
        y = y + 0
        print(X.shape)


    elif dataset_name in  ['OS1_Feature_Envy','OS1_Feature_Envy2','OS1_Feature_Envy3','OS1_Feature_Envy4']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('is_feature_envy').values
        if dataset_name in ['OS1_Feature_Envy','OS1_Feature_Envy2']:
           df = remove_unwanted_attributes_method(df)

        X = preprocessing_(df)
        y = y + 0
        print(X.shape)

    elif dataset_name in ['OS1_Long_Method','OS1_Long_Method2']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('is_long_method').values
        if dataset_name == 'OS1_Long_Method':
            df = remove_unwanted_attributes_method(df)
        X = preprocessing_(df)
        y = y + 0
        print(X.shape)

    elif dataset_name in ['OS2_ArgoUML_Functional_Decomposition','OS2_ArgoUML_Functional_Decomposition2',
                          'OS2_Azureus_Functional_Decomposition','OS2_Azureus_Functional_Decomposition2', 'OS2_Azureus_Functional_Decomposition23',
                          'OS2_Xerces_Functional_Decomposition','OS2_Xerces_Functional_Decomposition2','OS2_Xerces_Functional_Decomposition3',
                          'OS2_Functional_Decomposition', 'OS2_Functional_Decomposition2']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('FD').values
        X = preprocessing_(df)
        print(X.shape)

    elif dataset_name in ['OS2_ArgoUML_God_Class','OS2_ArgoUML_God_Class2',
                          'OS2_Azureus_God_Class', 'OS2_Azureus_God_Class2',
                          'OS2_Xerces_God_Class', 'OS2_Xerces_God_Class2',
                          'OS2_God_Class', 'OS2_God_Class2']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('BLOB').values
        X = preprocessing_(df)
        print(X.shape)

    elif dataset_name in ['OS2_ArgoUML_Spaghetti_Code','OS2_ArgoUML_Spaghetti_Code2',
                          'OS2_Azureus_Spaghetti_Code',  'OS2_Azureus_Spaghetti_Code2',
                          'OS2_Xerces_Spaghetti_Code','OS2_Xerces_Spaghetti_Code2',
                          'OS2_Spaghetti_Code', 'OS2_Spaghetti_Code2']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('SC').values
        X = preprocessing_(df)
        print(X.shape)

    elif dataset_name in ['OS2_ArgoUML_Swiss_Army_Knife','OS2_ArgoUML_Swiss_Army_Knife2',
                          'OS2_Azureus_Swiss_Army_Knife', 'OS2_Azureus_Swiss_Army_Knife2',
                          'OS2_Xerces_Swiss_Army_Knife','OS2_Xerces_Swiss_Army_Knife2',
                          'OS2_Swiss_Army_Knife', 'OS2_Swiss_Army_Knife2','OS2_Xerces_Swiss_Army_Knife3']:
        df = pd.read_csv(directory + dataset_name + '.csv')
        y = df.pop('SAK').values
        X = preprocessing_(df)
        print(X.shape)


    # -----------------End selecting dataset---------------------------------

    else:
        print("dataset %s does not exist" % dataset_name)


    return np.array(X), np.array(y), []



