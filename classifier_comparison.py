import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
import datetime
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
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#loading dataset
df= pd.read_csv('OS1_Data_Class.csv')
Y_output = df.pop('is_data_class').values
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
X_input = mean_imputer.transform(df.values)
scaler=MinMaxScaler(feature_range=(0,1))
X_input=scaler.fit_transform(X_input)
Y_output = Y_output + 0 
#___________________________________________________

# load dataset
dataset = "OS1_Data_Class.csv"
#X_input, Y_output = load_dataset(dataset)

# prepare models
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


# evaluate each model in turn
names = []

accuracyresults = []
accuracy = []

aucresults = []
auc = []

f1results = []
f1 = []

precisionresults = []
precision = []

recallresults = []
recall = []

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