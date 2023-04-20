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

dataset_name = 'PandasDataframes/Recorded_Melspecs_16.csv'

data = pd.read_csv(dataset_name).drop(['Unnamed: 0'], axis='columns').drop(index=779)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='LABEL')**(1/2), data.LABEL, test_size=0.1)


n_neighbors = 3

class_weights = [1,1,1,1,1]
#class_weights = [2.24,12.68,4.34,4.34,4.34] #weights adjusted manually

run_MLP = True

weights_list = [class_weights[int(i)] for i in y_train]

my_Normalizer = Normalizer() #Normalizer(norm='l2')

def my_kfold(X, y, model='xgb', display=False, n_splits=5) :
    my_kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    kf_scores = []
    for i, (train_index, test_index) in enumerate(my_kf.split(X, y)):
        weights_list = [class_weights[int(i)] for i in y.iloc[train_index]]
        if model == 'xgb' :
            XGB_model.fit(X.iloc[train_index], y.iloc[train_index], **{'XGBClassifier__sample_weight': weights_list})
            y_pred = XGB_model.predict(X.iloc[test_index])
        elif model == 'knn' :
            knn_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = knn_model.predict(X.iloc[test_index])
        elif model == 'mlp' :
            MLP_model.fit(X.iloc[train_index], y.iloc[train_index])
            y_pred = MLP_model.predict(X.iloc[test_index])
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

my_MLP = neural_network.MLPClassifier(max_iter=500, 
                                      hidden_layer_sizes=(640,1280,640),
                                      learning_rate='adaptive',
                                      nesterovs_momentum=False,
                                      alpha=0.015)

#PART 1 : PCA DIMENSION OPTIMIZATION
#---------------------------------------------------------------------------------------------------------------------

all_n_pca = range(12,24)

knn_scores = []
xgb_scores = []
mlp_scores = []

with open('PCALog.txt', 'w') as file :
    file.write('Recording Different PCA scores with default parameter tuning \n\r')
    for n_pca in all_n_pca :
        file.write(f'\n Scores for PCA size {n_pca} :\n\r')
        my_pca = PCA(n_components=n_pca)
        knn_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('KNNClassifier', my_KNN)])
        knn_score = my_kfold(X_train, y_train, model='knn', n_splits=30)
        knn_scores.append(knn_score)
        file.write(f'Score for KNN : {knn_score} \n\r')
        XGB_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('XGBClassifier', my_XGB)])
        xgb_score = my_kfold(X_train, y_train, model='xgb', n_splits=30)
        xgb_scores.append(xgb_score)
        file.write(f'Score for XGB : {xgb_score} \n\r')
        MLP_model = Pipeline([('NORM', my_Normalizer), ('PCA', my_pca), ('MLPClassifier', my_MLP)])
        mlp_score = my_kfold(X_train, y_train, model='mlp', n_splits=30)
        mlp_scores.append(mlp_score)
        file.write(f'Score for MLP : {mlp_score} \n\r')
    best_knn_pca = all_n_pca[np.argmax(np.array(knn_scores))]
    best_xgb_pca = all_n_pca[np.argmax(np.array(xgb_scores))]
    best_MLP_pca = all_n_pca[np.argmax(np.array(mlp_scores))]  
    file.write(f'\n\n\rBest PCA for KNN : {best_knn_pca}\n\r')
    file.write(f'Best PCA for KNN : {best_xgb_pca}\n\r')
    file.write(f'Best PCA for MLP : {best_MLP_pca}\n\r')

knn_X_train = Pipeline([('NORM', my_Normalizer), ('PCA', PCA(n_components=best_knn_pca))]).fit_transform(X_train)
xgb_X_train = Pipeline([('NORM', my_Normalizer), ('PCA', PCA(n_components=best_xgb_pca))]).fit_transform(X_train)
mlp_X_train = Pipeline([('NORM', my_Normalizer), ('PCA', PCA(n_components=best_MLP_pca))]).fit_transform(X_train)

#PART 2 : HYPERPARAMETER OPTIMIZATION
#---------------------------------------------------------------------------------------------------------------------
'''
#PART 2.1 : KNN

KNN_params = {'n_neighbors' : list(range(1,20)),
              'weights' : ['uniform', 'distance'],
              'p' : [1,2,3,4]}

KNN_GS = GridSearchCV(KNeighborsClassifier(), KNN_params, n_jobs=-1, cv=10, scoring='accuracy')
KNN_GS.fit(knn_X_train, y_train)

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