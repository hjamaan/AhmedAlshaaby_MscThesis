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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier

#Defin the list of datasets 
datasets = ['OS1_Data_Class.csv','OS1_God_Class.csv', 'OS1_Long_Method.csv', 'OS1_Feature_Envy.csv','OS2_ArgoUML_Functional_Decomposition.csv', 'OS2_ArgoUML_God_Class.csv', 'OS2_ArgoUML_Spaghetti_Code.csv', 'OS2_ArgoUML_Swiss_Army_Knife.csv'
        ,'OS2_Azureus_Functional_Decomposition.csv','OS2_Azureus_God_Class.csv','OS2_Azureus_Spaghetti_Code.csv','OS2_Azureus_Swiss_Army_Knife.csv','OS2_Xerces_Functional_Decomposition.csv',
        'OS2_Xerces_God_Class.csv','OS2_Xerces_Spaghetti_Code.csv','OS2_Xerces_Swiss_Army_Knife.csv','OS2_Functional_Decomposition', 'OS2_God_Class', 'OS2_Spaghetti_Code', 'OS2_Swiss_Army_Knife']


# Defined  models
#--------------------------------------------
models = []
models.append(('RF', RandomForestClassifier(random_state = 42)))
models.append(('SVM', SVC(random_state = 42)))
models.append(('CART', DecisionTreeClassifier(random_state = 42)))
models.append(('LR', LogisticRegression(random_state = 42)))
models.append(('GBC', GradientBoostingClassifier(n_estimators=300,random_state = 42)))
models.append(('MNB', MultinomialNB(alpha=0.001)))
models.append(('MLP', MLPClassifier()))
models.append(('SGDC', SGDClassifier(random_state = 42)))
models.append(('GaussianP', GaussianProcessClassifier(random_state = 42)))
models.append(('BNB', BernoulliNB(alpha=0.001)))

               

#---------------------------
# Specify the K-fold Crossvalidation
num_folds = 10
seed = 7
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
df_list = []
#--------------------------------------------------------

# --------------Preprocessing dataset------------- 
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
  
#--------------End Preprocessing dataset-----------------------

#--------------Starting classifiers comparison-----------------
for dataset in datasets:
  Model_names=[]
  Accuracy_results=[]
  Accuracy=[]
  AUC_results=[]
  AUC=[]
  f1_results=[]
  f1=[]
  Precision_results=[]
  Precision=[]
  Recall_results=[]
  Recall=[]
#---------------------------------

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
    
  elif dataset in ['OS2_ArgoUML_Functional_Decomposition','OS2_ArgoUML_Functional_Decomposition2',
                          'OS2_Azureus_Functional_Decomposition','OS2_Azureus_Functional_Decomposition2',
                          'OS2_Xerces_Functional_Decomposition','OS2_Xerces_Functional_Decomposition2',
                          'OS2_Functional_Decomposition', 'OS2_Functional_Decomposition2']:
    y = df.pop('FD').values  
    X=preprocessing_(df)
  
  elif dataset in ['OS2_ArgoUML_God_Class','OS2_ArgoUML_God_Class2',
                          'OS2_Azureus_God_Class', 'OS2_Azureus_God_Class2',
                          'OS2_Xerces_God_Class', 'OS2_Xerces_God_Class2',
                          'OS2_God_Class', 'OS2_God_Class2']:
    y = df.pop('BLOB').values  
    X=preprocessing_(df)
 
  elif dataset in ['OS2_ArgoUML_Spaghetti_Code','OS2_ArgoUML_Spaghetti_Code2',
                          'OS2_Azureus_Spaghetti_Code',  'OS2_Azureus_Spaghetti_Code2',
                          'OS2_Xerces_Spaghetti_Code','OS2_Xerces_Spaghetti_Code2',
                          'OS2_Spaghetti_Code', 'OS2_Spaghetti_Code2']:
    y = df.pop('SC').values  
    X=preprocessing_(df)
  
  elif dataset in ['OS2_ArgoUML_Swiss_Army_Knife','OS2_ArgoUML_Swiss_Army_Knife2',
                          'OS2_Azureus_Swiss_Army_Knife', 'OS2_Azureus_Swiss_Army_Knife2',
                          'OS2_Xerces_Swiss_Army_Knife','OS2_Xerces_Swiss_Army_Knife2',
                          'OS2_Swiss_Army_Knife', 'OS2_Swiss_Army_Knife2']:
    y = df.pop('SAK').values  
    X=preprocessing_(df)
  else:
        print("dataset %s does not exist" % dataset)
#-----------------End selecting dataset---------------------------------
  print('----------------------------------------')
  print('----------------------------------------')
  print('----------------------------------------')
  print(dataset)
  print('----------------------------------------')
  print('----------------------------------------')
  print('----------------------------------------')
  
  for name, model in models:
    Model_names.append(name)
    cv_accuracy = model_selection.cross_val_score(model,  X, y, cv=kfold, scoring='accuracy')
    cv_auc = model_selection.cross_val_score(model,  X, y, cv=kfold, scoring='roc_auc')
    cv_prec = model_selection.cross_val_score(model,  X, y, cv=kfold, scoring='precision')
    cv_recall = model_selection.cross_val_score(model,  X, y, cv=kfold, scoring='recall')
    cv_f1 = model_selection.cross_val_score(model,  X, y, cv=kfold, scoring='f1')
    Accuracy_results.append(cv_accuracy)
    AUC_results.append(cv_auc)
    Precision_results.append(cv_prec)
    Recall_results.append(cv_recall)
    f1_results.append(cv_f1)
    msg="%s: %f (%f)"%(name, cv_accuracy.mean(), cv_accuracy.std())
    Accuracy.append(cv_accuracy.mean())
    AUC.append(cv_auc.mean())
    Precision.append(cv_prec.mean())
    Recall.append(cv_recall.mean())
    f1.append(cv_f1.mean())
    print('----------------------------------------')
    print(msg)
    Y_pred = cross_val_predict(model,X,y,cv=kfold)
    conf_mat = confusion_matrix(y,Y_pred)
    print(conf_mat)
    print('----------------------------------------')

  

#----------------boxplot for accuracy comparison-----
  graph = plt.figure()
  graph.suptitle('Accuracy Comparison')
  ax = graph.add_subplot(111)
  plt.boxplot(Accuracy_results)
  ax.set_xticklabels(Model_names)

  y_pos = np.arange(len(Accuracy))
  # bar chart accuracy comparison
  graph2 = plt.figure()
  graph2.suptitle('Accuracy Comparison')
  ax2 = graph2.add_subplot(111)
  plt.bar(y_pos, Accuracy, align='center', alpha=0.5)
  plt.xticks(y_pos, Model_names)
  plt.show()


  #Removing unwantd characters
  #----------------------------------------------------------
  Model_names = str(Model_names)
  Model_names = Model_names.replace('[','').replace(']','').replace("'","")
  Accuracy = str(Accuracy)
  Accuracy = Accuracy.replace('[','').replace(']','').replace("'","")
  AUC = str(AUC)
  AUC = AUC.replace('[','').replace(']','').replace("'","")
  Precision = str(Precision)
  Precision = Precision.replace('[','').replace(']','').replace("'","")
  Recall = str(Recall)
  Recall = Recall.replace('[','').replace(']','').replace("'","")
  f1 = str(f1)
  f1 = f1.replace('[','').replace(']','').replace("'","")
  #------------------------------------------------------
  
  i=1
  #Writing to csv
  file=open('Result'+' '+'of'+' '+dataset+'.csv', 'w')
  file.write(dataset +'\n')
  file.write(str("10-KFold") +'\n')
  file.write(' ,')
  file.write(str(Model_names))
  file.write('\nAccuracy,')
  file.write(str(Accuracy))
  file.write('\nAuc,')
  file.write(str(AUC))
  file.write('\nPrecision,')
  file.write(str(Precision))
  file.write('\nRecall,')
  file.write(str(Recall))
  file.write('\nf1,')
  file.write(str(f1))
  file.close()
  ++i
  #--------------End classifiers comparison-----------------

