import numpy as np
import pandas as pd

"Machine learning tools"
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import neural_network
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, Normalizer
import xgboost as xgb
import pickle

from utils_copy import *

test_number = 0

#just adapt the dataset name : since it is generated with Dataframe.to_csv(), we need to remove the column ['Unnamed: 0']
dataset_name = 'PandasDataframes/BigData_PCA_Normalized.csv'

ESC50_data = pd.read_csv(dataset_name).drop(['Unnamed: 0'], axis='columns')
n = ESC50_data.shape[0]

#Xtrain will then be further split
X_glob, X_save, y_glob, y_save = train_test_split(ESC50_data.drop(columns='LABEL'), ESC50_data.LABEL, test_size=0.1)
n = X_glob.shape[0]
my_Normalizer = Normalizer() 

#TRAINING A KNN CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

my_KNN = KNeighborsClassifier(n_neighbors=3, p=3, weights='distance')

KNN_model = Pipeline([('KNN',my_KNN)])

#TRAINING A XGB CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

my_XGB = xgb.sklearn.XGBClassifier(max_depth=12,min_child_weight=0,n_estimators=1500)

XGB_model = Pipeline([('Normalizer', my_Normalizer), ('XGBClassifier', my_XGB)])

#TRAINING A MLP CLASSIFIER
#-----------------------------------------------------------------------------------------------------------

my_MLP = neural_network.MLPClassifier(hidden_layer_sizes=(640,1280,640), max_iter=800, solver='adam',
                                      alpha=0.015, learning_rate='adaptive', nesterovs_momentum=False)

MLP_model = Pipeline([('Normalizer', my_Normalizer), ('MLPClassifier', my_MLP)])

#TESTING THE MODEL ON DATA FROM THE MICROPHONE
#----------------------------------------------------------------------------------------------------------

all_files = [open('knn_log.txt', 'w'), open('xgb_log.txt', 'w'), open('mlp_log.txt', 'w'), open('mixed_log.txt', 'w')]

def my_kfold(X, y, display=False, plus=True, a=None) :
    m = 5
    for file in all_files :
        file.write(f'Running test {test_number} with {m} splits, {X.shape} dataset : \n')
        
    my_kf = StratifiedKFold(n_splits=m, shuffle=True)
    knn_scores = []
    xgb_scores = []
    mlp_scores = []
    mixed_scores = []
    for i, (train_index, test_index) in enumerate(my_kf.split(X, y)):
        XGB_model.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = XGB_model.predict(X.iloc[test_index])
        knn_scores.append(get_accuracy(y_pred, y.iloc[test_index]))
        KNN_model.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = KNN_model.predict(X.iloc[test_index])
        xgb_scores.append(get_accuracy(y_pred, y.iloc[test_index]))
        MLP_model.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = MLP_model.predict(X.iloc[test_index])
        mlp_scores.append(get_accuracy(y_pred, y.iloc[test_index]))
        if plus:
            y_mixed_proba = KNN_model.predict_proba(X.iloc[test_index])*0.1 + XGB_model.predict_proba(X.iloc[test_index])*0.9 + MLP_model.predict_proba(X.iloc[test_index])*0.9
        else :
            y_mixed_proba = KNN_model.predict_proba(X.iloc[test_index])**a[0]*XGB_model.predict_proba(X.iloc[test_index])**a[1]*MLP_model.predict_proba(X.iloc[test_index])**a[2]
        y_mixed = [np.argmax(np.array(probas)) for probas in y_mixed_proba]
        mixed_scores.append(get_accuracy(y_mixed, y.iloc[test_index]))

    if display : print(f'K-fold accuracy : {sum(mixed_scores)/len(mixed_scores)}')
    all_files[0].write(str(sum(knn_scores)/len(mixed_scores))+'\n')
    all_files[1].write(str(sum(xgb_scores)/len(mixed_scores))+'\n')
    all_files[2].write(str(sum(mlp_scores)/len(mixed_scores))+'\n')
    all_files[3].write(str(sum(mixed_scores)/len(mixed_scores))+'\n')
    return sum(mixed_scores)/len(mixed_scores)

dataset_sizes = [50,100,200,300,500,700,1000,1300,1800]

weights = np.ones((9,3))
for i in range(2) :
    np.random.seed(1107)
    for j in range(4) :
        weights[4*i+j+1] = np.random.random(3)+1-i

for size in dataset_sizes :
    X_train, X_test, y_train, y_test = train_test_split(X_glob, y_glob, test_size=(n-size)/n)
    my_kfold(X_train, y_train)
    if size == 7000 :
        for a in weights :
            my_kfold(X_train, y_train, plus=False, a=a)


#TRAINING MODELS ON ENTIRE DATASET AND MAKING PICKLE FILES
#----------------------------------------------------------------------------------------------------------
if False :
    X, y = ESC50_data.drop(columns=['LABEL']), ESC50_data.LABEL

    with open(file = 'Models/KNN_model_noPCA.pickle', mode = 'wb') as file :
        KNN_model.fit(X,y)
        pickle.dump(KNN_model, file)

    with open(file = 'Models/XGB_model_noPCA.pickle', mode = 'wb') as file :
        XGB_model.fit(X,y)
        pickle.dump(XGB_model, file)

    with open(file = 'Models/MLP_model_noPCA.pickle', mode = 'wb') as file :
        MLP_model.fit(X,y)
        pickle.dump(MLP_model, file)
