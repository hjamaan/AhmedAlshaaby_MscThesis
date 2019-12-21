import utilities as ut
import numpy as np
from sklearn.svm import SVC
import pylab as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score,classification_report,confusion_matrix
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression    
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.gaussian_process import GaussianProcessClassifier
def run_method(method,dataset_name,X, y, n_clfs=10, score_name="auc"):
    if method == "features_selection":

        w_svm = SVC(class_weight='balanced', probability=True)
       # w_svm1=RandomForestClassifier()

    if method =="APE_normal":


         fss = ["ensemble_svm", "ensemble_heter"]

         for fsss in fss:
            # scores, x_values= 0,0
             if fsss =="ensemble_svm":
               clfs=[]
               for c in [1, 10, 100, 500, 1000]:
                   for w in [{1: 5}, {1: 10}, {1: 15}, {1: 20}, {1: 25}]:
                       clfs += [SVC(probability=True, C=c, class_weight=w)]

               (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)
               if fsss=="ensemble_svm":
                  plt.plot(x_values, scores, label="Normal Homogeneous Ensemble")




               plt.xlabel("Number of Classifiers")



             elif fsss =="ensemble_heter":


                scaler = MinMaxScaler(feature_range=(0, 1))
                X = scaler.fit_transform(X)

                clfs = [SVC(probability=True), MultinomialNB(alpha=0.001),
                 BernoulliNB(alpha=0.001), RandomForestClassifier(n_estimators=20),
                 GradientBoostingClassifier(n_estimators=300),
                 SGDClassifier(alpha=.0001, loss='log', n_iter=50,
                 penalty="elasticnet"), LogisticRegression(penalty='l2'),
                 GaussianProcessClassifier(),DecisionTreeClassifier(), MLPClassifier(),SVC(probability=True), MultinomialNB(alpha=0.001)]

                (scores, x_values) = ensemble_forward_pass(clfs, X, y, n_clfs=n_clfs)

                plt.plot(x_values, scores, label="Normal Heterogeneous Ensemble")

                plt.xlabel("Number of Classifiers")


    else:
        print("%s does not exist..." % method)


#### ENSEMBLE FORWARD PASS
def ensemble_forward_pass(clfs, X, y, n_clfs=10):
    if n_clfs == None:
        n_clfs= len(clfs)

    clf_list = ut.ensemble_clfs(clfs)
    auc_scores = np.zeros(n_clfs)

    for i in range(n_clfs):
        skf = model_selection.StratifiedKFold(n_splits=10)

        # CROSS VALIDATE

        f= "auc"
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf_list.fit(X_train, y_train, i)
            y_pred = clf_list.predict(X_test)

            if f=="auc" :
                scores += [metrics.roc_auc_score(y_test, y_pred)]




        auc_scores[i] = np.mean(scores)
        print("Score: %.3f, n_clfs: %d" % (auc_scores[i], i+1))

    return auc_scores, np.arange(n_clfs) + 1

