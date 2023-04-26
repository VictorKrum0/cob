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

data = pd.read_csv(dataset_name).drop(['Unnamed: 0'], axis='columns')

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='LABEL'), data.LABEL, test_size=0.1)

n_neighbors = 3

class_weights = [1,1,1,1,1]

my_Normalizer = Normalizer() #Normalizer(norm='l2')

def my_kfold(X, y, display=False, n_splits=5, w = np.array([0.33,0.33,0.33])) :
    my_kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    single_model_scores = [[],[],[]]
    kf_scores = []
    for i, (train_index, test_index) in enumerate(my_kf.split(X, y)):
        XGB_model.fit(X.iloc[train_index], y.iloc[train_index])
        single_model_scores[0].append(get_accuracy(KNN_model.predict(X.iloc[test_index],y.iloc[test_index])))
        KNN_model.fit(X.iloc[train_index], y.iloc[train_index])
        single_model_scores[1].append(get_accuracy(XGB_model.predict(X.iloc[test_index]),y.iloc[test_index]))
        MLP_model.fit(X.iloc[train_index], y.iloc[train_index])
        single_model_scores[2].append(get_accuracy(MLP_model.predict(X.iloc[test_index]),y.iloc[test_index]))
        y_mixed_proba = KNN_model.predict_proba(X.iloc[test_index])*w[0] + XGB_model.predict_proba(X.iloc[test_index])*w[1] + MLP_model.predict_proba(X.iloc[test_index])*w[2]
        y_mixed = [np.argmax(np.array(probas)) for probas in y_mixed_proba] 
        kf_scores.append(get_accuracy(y_mixed, y.iloc[test_index]))
    if display : print(f'K-fold accuracy : {sum(kf_scores)/len(kf_scores)}')
    return sum(kf_scores)/len(kf_scores), single_model_scores

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

KNN_model = Pipeline([('NORM', my_Normalizer), ('KNNClassifier', my_KNN)])
XGB_model = Pipeline([('NORM', my_Normalizer), ('XGBClassifier', my_XGB)])
MLP_model = Pipeline([('NORM', my_Normalizer), ('MLPClassifier', my_MLP)])

#add a way to compute a single model and then test it with all weight combinations
#put in a lot of folds (50)
weights_list = np.ones((13,3))
for i in range(2) :
    np.random.seed(1107)
    for j in range(6) :
        weights_list[6*i+j+1] = np.random.random(3)+1-i

with open('ModelComparisonReport.txt', 'w') as file :
    file.write('Report for running the mixed model with different weighting coefficients \n')
    for weights in weights_list :
        scores, single_model_scores = my_kfold(X_train,y_train,n_splits=20,w=weights)
        knn_score = np.mean(np.array(single_model_scores[0]))
        xgb_score = np.mean(np.array(single_model_scores[1]))
        mlp_score = np.mean(np.array(single_model_scores[2]))
        file.write(f'\n\n Score for weights {weights} : {scores}. \n\n Single model performance on CV :\n')
        file.write(f'KNN : {knn_score}\nXGB : {xgb_score}\nMLP : {mlp_score}\n')