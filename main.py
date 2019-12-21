import sys
import argparse
import utilities as ut
import numpy as np
import pylab as pl
import methods
import warnings
from sklearn.preprocessing import MinMaxScaler,Imputer
from sklearn.decomposition import PCA
if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    # fs_functions defines the forward selection functions

    datasets = ['OS1_Data_Class', 'OS1_God_Class', 'OS1_Feature_Envy', 'OS1_Long_Method',
                'OS2_ArgoUML_Functional_Decomposition', 'OS2_ArgoUML_God_Class',
                'OS2_ArgoUML_Spaghetti_Code', 'OS2_ArgoUML_Swiss_Army_Knife',
                'OS2_Azureus_Functional_Decomposition', 'OS2_Azureus_God_Class',
                'OS2_Azureus_Spaghetti_Code','OS2_Azureus_Swiss_Army_Knife',
                'OS2_Xerces_Functional_Decomposition',  'OS2_Xerces_God_Class',
                'OS2_Xerces_Spaghetti_Code', 'OS2_Xerces_Swiss_Army_Knife',
                'OS2_Functional_Decomposition', 'OS2_God_Class', 'OS2_Spaghetti_Code', 'OS2_Swiss_Army_Knife']

    parser.add_argument('-m', '--method', default="APE_normal",
                        choices=["APE_normal",
                                  "ensemble_heter",
                                 "ensemble_svm"
                               ])

    parser.add_argument('-d', '--dataset_name', default="OS1_Data_Class")

    parser.add_argument('-n', '--n_clfs', default=10, type=int)

    parser.add_argument('-s', '--score_name', default="auc",
                        choices=["auc"])

    args = parser.parse_args()
    method = args.method
    dataset_name = args.dataset_name
    
    n_clfs = args.n_clfs
    score_name = args.score_name
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print("\nDATASET: %s\nMETHOD: %s\n" % (dataset_name, method))
    np.random.seed(1)
    ##### 1. ------ GET DATASET
    X, y, ft_names = ut.read_dataset("datasets/", dataset_name=dataset_name)
    pl.title(dataset_name)
    pl.ylabel("AUC")

    ##### 2. ------- RUN TRANING METHOD
    methods.run_method(method,dataset_name, X, y, n_clfs=n_clfs,

                       score_name=score_name)

    pl.legend(loc="lower right")
    pl.show()




