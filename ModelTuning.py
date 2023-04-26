import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

"Machine learning tools"
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize, Normalizer, StandardScaler
import xgboost as xgb

from utils_copy import *

warnings.filterwarnings("ignore")

dataset_name = 'PandasDataframes/BigData_PCA.csv'

data = pd.read_csv(dataset_name).drop(['Unnamed: 0'], axis='columns').drop(index=779)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='LABEL')**(1/2), data.LABEL, test_size=0.1)

n_neighbors = 3

class_weights = [1,1,1,1,1]

my_Normalizer = Normalizer() #Normalizer(norm='l2')

def my_kfold(X, y, display=False, n_splits=5) :
    my_kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    kf_scores = []
    for i, (train_index, test_index) in enumerate(my_kf.split(X, y)):
        weights_list = [class_weights[int(i)] for i in y.iloc[train_index]]
        XGB_model.fit(X.iloc[train_index], y.iloc[train_index], **{'XGBClassifier__sample_weight': weights_list})
        KNN_model.fit(X.iloc[train_index], y.iloc[train_index])
        MLP_model.fit(X.iloc[train_index], y.iloc[train_index])
        y_mixed_proba = KNN_model.predict_proba(X.iloc[test_index])/3 + XGB_model.predict_proba(X.iloc[test_index])/3 + MLP_model.predict_proba(X.iloc[test_index])/3
        y_mixed = [np.argmax(np.array(probas)) for probas in y_mixed_proba] 
        kf_scores.append(get_accuracy(y_mixed, y.iloc[test_index]))
    if display : print(f'K-fold accuracy : {sum(kf_scores)/len(kf_scores)}')
    return sum(kf_scores)/len(kf_scores)

def my_kfold_xgb(X, y, display=False, n_splits=5, class_weights=[1,1,1,1,1]) :
    my_kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    kf_scores = []
    for i, (train_index, test_index) in enumerate(my_kf.split(X, y)):
        weights_list = [class_weights[int(i)] for i in y.iloc[train_index]]
        XGB_model.fit(X.iloc[train_index], y.iloc[train_index], **{'XGBClassifier__sample_weight': weights_list})
        y_pred = XGB_model.predict(X.iloc[test_index])
        kf_scores.append(get_accuracy(y_pred, y.iloc[test_index]))
    if display : print(f'K-fold accuracy : {sum(kf_scores)/len(kf_scores)}')
    return sum(kf_scores)/len(kf_scores)

#TRAINING A KNN CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

my_KNN = KNeighborsClassifier(n_neighbors=n_neighbors,weights='distance',p=3)

#TRAINING A XGB CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

my_XGB = xgb.sklearn.XGBClassifier(max_depth=16, n_estimators=1600, min_child_weight=0)
#XGB_model.fit(X_train, y_train, **{'XGBClassifier__sample_weight': weights_list})

#TRAINING A MLP CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

my_MLP = neural_network.MLPClassifier(max_iter=800, 
                                      hidden_layer_sizes=(640,1280,640),
                                      learning_rate='adaptive',
                                      nesterovs_momentum=False,
                                      alpha=0.015)

#PART 1 : PCA DIMENSION OPTIMIZATION
#---------------------------------------------------------------------------------------------------------------------
all_n_pca = range(12,13)

Mixed_scores = []

with open('PCALog.txt', 'w') as file :
    file.write('Recording Different PCA scores with default parameter tuning \n\r')
    for n_pca in all_n_pca :
        file.write(f'\n Score for PCA size {n_pca} :\n\r')
        my_pca = PCA(n_components=n_pca)
        KNN_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('KNNClassifier', my_KNN)])
        XGB_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('XGBClassifier', my_XGB)])
        MLP_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('MLPClassifier', my_MLP)])
        mixed_score = my_kfold(X_train, y_train, n_splits=20)
        Mixed_scores.append(mixed_score)
        file.write(f'Score for Mixed : {mixed_score} \n\r')
    best_pca = all_n_pca[np.argmax(np.array(Mixed_scores))]
    file.write(f'\n\n\rBest PCA for Mixed : {best_pca}\n\r')

my_pca = PCA(n_components=best_pca)
print(best_pca)

KNN_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('KNNClassifier', my_KNN)])
XGB_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('XGBClassifier', my_XGB)])
MLP_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('MLPClassifier', my_MLP)])

#PART 2 : HYPERPARAMETER OPTIMIZATION
#---------------------------------------------------------------------------------------------------------------------

class_weights = [[1,1,1,1,1],[2.24,12.68,4.34,4.34,4.34],[1,3,1.5,1,1],[1,10,5,5,5],[2,10,5,5,5],[1,10,5,4,3]]

weights_scores = []

with open('WeightScores.txt','w') as file :
    for class_weights_i in class_weights :
        score = my_kfold_xgb(X_train, y_train, n_splits=30, class_weights=class_weights_i)
        file.write(f'Score for {class_weights_i} : {score} \n\r')
        weights_scores.append(score)
    file.write(f'\nBest score was obtained for {class_weights[np.argmax(np.array(weights_scores))]}')

'''
#PART 2.1 : KNN

KNN_params = {'n_neighbors' : list(range(1,20)),
              'weights' : ['uniform', 'distance'],
              'p' : [1,2,3,4]}

KNN_GS = GridSearchCV(KNeighborsClassifier(), KNN_params, n_jobs=-1, cv=10, scoring='accuracy')
KNN_GS.fit(KNN_X_train, y_train)

#PART 2.2 : XGB

XGB_params = {
        'max_depth': [10,13,14,15,16,17,18,19,20,21,22,23,25,30,50],
        'n_estimators' : [800,900,1000,1100,1200,1500,2000],
        'min_child_weight' : [0,0.5,1]
    }

XGB_GS = GridSearchCV(xgb.XGBClassifier(),
                      XGB_params, n_jobs=-1, cv=10, scoring='accuracy')
XGB_GS.fit(xgb_X_train, y_train)

#PART 2.3 : MLP

MLP_params = {
        'hidden_layer_sizes': [(320,640,320),(640,640,640),(640,1280,640)],
        'solver': ['adam'],
        'alpha': [0.01, 0.0075, 0.015],
        'momentum' : [0,0.65,0.7,0.75],
        'learning_rate': ['constant','adaptive']
    }

MLP_GS = GridSearchCV(neural_network.MLPClassifier(max_iter=800, nesterovs_momentum=True ,
                                                   activation='relu', early_stopping=True,
                                                   n_iter_no_change=50),
                      MLP_params, n_jobs=-1, cv=10, scoring='accuracy')
MLP_GS.fit(mlp_X_train, y_train)

with open('GridSearchLog.txt', 'w') as file :
    file.write(f'Results for grid searches on all selected models \n\r')
    #file.write(f'Best set of parameters for KNN model : {KNN_GS.best_params_} gave score {KNN_GS.best_score_}\n\r')
    #file.write(f'Best set of parameters for XGB model : {XGB_GS.best_params_} gave score {XGB_GS.best_score_}\n\r')
    file.write(f'Best set of parameters for MLP model : {MLP_GS.best_params_} gave score {MLP_GS.best_score_}\n\r')
'''