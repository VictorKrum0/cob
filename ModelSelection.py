import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"Machine learning tools"
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, Normalizer, StandardScaler
import xgboost as xgb
import pickle

from utils_copy import *

dataset_name = 'PandasDataframes/BigData.csv'

data = pd.read_csv(dataset_name).drop(['Unnamed: 0'], axis='columns').drop(index=779)

n_pca = 16

run_K_NN = True
n_neighbors = 3
pickle_KNN = False

run_Xgb = True
class_weights = [1,1,1,1,1]
#class_weights = [2.24,12.68,4.34,4.34,4.34] #weights adjusted manually
pickle_XGB = False

run_RF = True
pickle_RF = False

run_SVC = True
pickle_SVC = False

run_MLP = True
pickle_MLP = False

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='LABEL'), data.LABEL, test_size=0.1)

weights_list = [class_weights[int(i)] for i in y_train]

my_Normalizer = Normalizer() #Normalizer(norm='l2')
my_pca = PCA(n_components=n_pca)

def my_kfold(X, y, model='xgb', display=False) :
    my_kf = StratifiedKFold(n_splits=20, shuffle=True)
    kf_scores = []
    for i, (train_index, test_index) in enumerate(my_kf.split(X, y)):
        weights_list = [class_weights[int(i)] for i in y.iloc[train_index]]
        if model == 'xgb' :
            XGB_model.fit(X.iloc[train_index], y.iloc[train_index], **{'XGBClassifier__sample_weight': weights_list})
            y_pred = XGB_model.predict(X.iloc[test_index])
        elif model == 'knn' :
            knn_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = knn_model.predict(X.iloc[test_index])
        elif model == 'rf' :
            RF_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = RF_model.predict(X.iloc[test_index])
        elif model == 'svc' :
            SVC_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = SVC_model.predict(X.iloc[test_index])
        elif model == 'mlp' :
            MLP_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = MLP_model.predict(X.iloc[test_index])
        kf_scores.append(get_accuracy(y_pred, y.iloc[test_index]))
    if display : print(f'K-fold accuracy : {sum(kf_scores)/len(kf_scores)}')
    return sum(kf_scores)/len(kf_scores)

#TRAINING A KNN CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

if run_K_NN :
    my_KNN = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('KNN',my_KNN)])

    knn_model.fit(X_train, y_train)

#TRAINING A XGB CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

if run_Xgb :
    my_XGB = xgb.sklearn.XGBClassifier()

    XGB_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('XGBClassifier', my_XGB)])

    XGB_model.fit(X_train, y_train, **{'XGBClassifier__sample_weight': weights_list})

#TRAINING A RANDOM FOREST CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

if run_RF :
    my_RF = ensemble.RandomForestClassifier()

    RF_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('RandomForestClassifier', my_RF)])

    RF_model.fit(X_train, y_train)

#TRAINING A SVM CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

if run_SVC :
    my_SVC = svm.SVC()

    SVC_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('SVCClassifier', my_SVC)])

    SVC_model.fit(X_train, y_train)


#TRAINING A MLP CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

if run_MLP :
    my_MLP = neural_network.MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, solver='sgd')

    MLP_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('MLPClassifier', my_MLP)])

    MLP_model.fit(X_train, y_train)


#RUNNING CLASSIFIERS AND PICKLES
#----------------------------------------------------------------------------------------------------------

if run_K_NN :
    y_pred = knn_model.predict(X_test)
    print(f'Test set accuracy for K-NN classifier on Mic data : {get_accuracy(y_pred, y_test)} for {len(y_pred)} data items')
    show_confusion_matrix(y_pred, y_test, classnames=['CrackFire','ChirpBirds','Helicopter','Chainsaw','Handsaw'])
    my_kfold(X_train, y_train, model='knn', display=True)

if pickle_KNN :
    with open(file = 'KNN_model.pickle', mode = 'wb') as file :
        pickle.dump(knn_model, file)

if run_Xgb :
    y_pred = XGB_model.predict(X_test)
    print(f'Test Set Accuracy for XGB classifier on Mic data : {get_accuracy(y_pred, y_test)} for {len(y_pred)} data items')
    show_confusion_matrix(y_pred,y_test, classnames=['CrackFire','ChirpBirds','Helicopter','Chainsaw','Handsaw'])
    my_kfold(X_train, y_train, model='xgb', display=True)

if pickle_XGB :
    with open(file = 'XGB_model.pickle', mode = 'wb') as file :
        pickle.dump(XGB_model, file)

if run_RF :
    y_pred = RF_model.predict(X_test)
    print(f'Test Set Accuracy for Random Forest classifier on Mic data : {get_accuracy(y_pred, y_test)} for {len(y_pred)} data items')
    show_confusion_matrix(y_pred,y_test, classnames=['CrackFire','ChirpBirds','Helicopter','Chainsaw','Handsaw'])
    my_kfold(X_train, y_train, model='rf', display=True)

if pickle_RF :
    with open(file = 'RF_model.pickle', mode = 'wb') as file :
        pickle.dump(RF_model, file)

if run_SVC :
    y_pred = SVC_model.predict(X_test)
    print(f'Test Set Accuracy for SVM classifier on Mic data : {get_accuracy(y_pred, y_test)} for {len(y_pred)} data items')
    show_confusion_matrix(y_pred,y_test, classnames=['CrackFire','ChirpBirds','Helicopter','Chainsaw','Handsaw'])
    my_kfold(X_train, y_train, model='svc', display=True)

if pickle_SVC :
    with open(file = 'svc_model.pickle', mode = 'wb') as file :
        pickle.dump(SVC_model, file)

if run_MLP :
    y_pred = MLP_model.predict(X_test)
    print(f'Test Set Accuracy for MLP classifier on Mic data : {get_accuracy(y_pred, y_test)} for {len(y_pred)} data items')
    show_confusion_matrix(y_pred,y_test, classnames=['CrackFire','ChirpBirds','Helicopter','Chainsaw','Handsaw'])
    my_kfold(X_train, y_train, model='mlp', display=True)

if pickle_MLP :
    with open(file = 'MLP_model.pickle', mode = 'wb') as file :
        pickle.dump(MLP_model, file)
