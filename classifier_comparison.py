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
 
	cv_accuracy = model_selection.cross_val_score(model,  X_input, Y_output, cv=kfold, scoring='accuracy')
	cv_auc = model_selection.cross_val_score(model,  X_input, Y_output, cv=kfold, scoring='roc_auc')
	cv_prec = model_selection.cross_val_score(model,  X_input, Y_output, cv=kfold, scoring='precision')
	cv_recall = model_selection.cross_val_score(model,  X_input, Y_output, cv=kfold, scoring='recall')
	cv_f1 = model_selection.cross_val_score(model,  X_input, Y_output, cv=kfold, scoring='f1')
	accuracyresults.append(cv_accuracy)
	aucresults.append(cv_auc)
	precisionresults.append(cv_prec)
	recallresults.append(cv_recall)
	f1results.append(cv_f1)
	
	msg = "%s: %f (%f)" % (name, cv_accuracy.mean(), cv_accuracy.std())
	accuracy.append(cv_accuracy.mean())
	auc.append(cv_auc.mean())
	precision.append(cv_prec.mean())
	recall.append(cv_recall.mean())
	f1.append(cv_f1.mean())
	
	
	print('----------------------------------------')
	print(msg)
	Y_pred = cross_val_predict(model,X_input,Y_output,cv=kfold)
	conf_mat = confusion_matrix(Y_output,Y_pred)
	print(conf_mat)
	print('----------------------------------------')


# boxplot for accuracy comparison
graph = plt.figure()
graph.suptitle('Accuracy Comparison')
ax = graph.add_subplot(111)
plt.boxplot(accuracyresults)
ax.set_xticklabels(names)

y_pos = np.arange(len(accuracy))
# bar chart accuracy comparison
graph2 = plt.figure()
graph2.suptitle('Accuracy Comparison')
ax2 = graph2.add_subplot(111)
plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, names)
plt.show()


#Removing unwantd characters
#----------------------------------------------------------
names = str(names)
names = names.replace('[','').replace(']','').replace("'","")
accuracy = str(accuracy)
accuracy = accuracy.replace('[','').replace(']','').replace("'","")
auc = str(auc)
auc = auc.replace('[','').replace(']','').replace("'","")
precision = str(precision)
precision = precision.replace('[','').replace(']','').replace("'","")
recall = str(recall)
recall = recall.replace('[','').replace(']','').replace("'","")
f1 = str(f1)
f1 = f1.replace('[','').replace(']','').replace("'","")
#------------------------------------------------------

#Writing to csv
file=open('E:\kfupm\result1.csv', 'w+')
file.write(dataset +'\n')
file.write(str("10-KFold") +'\n')
file.write(' ,')
file.write(str(names))
file.write('\nAccuracy,')
file.write(str(accuracy))
file.write('\nAuc,')
file.write(str(auc))
file.write('\nPrecision,')
file.write(str(precision))
file.write('\nRecall,')
file.write(str(recall))
file.write('\nf1,')
file.write(str(f1))
file.close()
