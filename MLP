from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
 from sklearn.metricsicty import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
df= pd.read_csv('OS1_Data_Class.csv')
Y = df.pop('is_data_class').values
df.pop('IDType')
df.pop('project')
df.pop('package')
df.pop('complextype')
X = np.array(df)
#missing values
X[X == '?'] = -1
X[X == 'nan'] = -1
X = X.astype('float')
#Rescaling data
scaler=MinMaxScaler(feature_range=(0,1))
X=scaler.fit_transform(X)
#-------------------
#conert lables to 0 or 1
Y = Y + 0 
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
def create_model():
  model = Sequential()
  model.add(Dense(8, input_dim=61, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(8, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
                metrics=['accuracy'])
  return model
 
def specif_score(y_true, y_pred):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  specificity = tn / (tn+fp)
  return specificity
 #finding accuracy and f1 m 
 cvscores = {'acc': [] , 'f1': []}

for train, test in kfold.split(X, Y):
  model = create_model()
  model.fit(X[train], Y[train], epochs=200, batch_size=10, verbose=0)
  # evaluate the model
  #scores = model.evaluate(X[test], Y[test], verbose=0)
  #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  #cvscores.append(scores[1] * 100)
  predicted = model.predict(X)
  predicted[predicted > 0.5]  = 1
  predicted[predicted <= 0.5] = 0 
  cvscores['f1'].append(f1_score(Y, predicted))
  cvscores['acc'].append(accuracy_score(Y, predicted))
  print(f"acc: {np.mean(cvscores['acc'])}, f1: {np.mean(cvscores['f1'])}")

# printing roc 
scores = {'auc':[], 'spec':[], 'sens':[]}

for thresh in np.linspace(0, 1, num = 10):
  print(f'threshold {thresh}')
  auc = []
  sens = []
  spec = []
  
  for train, test in kfold.split(X, Y):
    model = create_model()
    model.fit(X[train], Y[train], epochs=10, batch_size=10, verbose=0)
    # evaluate the model
    #scores = model.evaluate(X[test], Y[test], verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #cvscores.append(scores[1] * 100)
    predicted = model.predict(X)
    predicted[predicted > thresh]  = 1
    predicted[predicted <= thresh] = 0 
    auc.append( roc_auc_score(Y, predicted))
    sens.append( recall_score(Y, predicted))
    spec.append( specificty_score(Y, predicted))
  scores['auc'].append(np.mean(auc))
  scores['spec'].append(np.mean(sens))
  scores['sens'].append(np.mean(spec))
  #dowing the roc
  
x = 1-  np.array(scores['spec'])
y = np.array(scores['sens'])
plt.plot(x, y )
