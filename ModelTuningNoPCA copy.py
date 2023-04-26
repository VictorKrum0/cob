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

weights = np.ones((13,3))
for i in range(2) :
    np.random.seed(1107)
    for j in range(6) :
        weights[6*i+j+1] = np.random.random(3)+1-i

print(weights)

np.random.seed(1107)

print(np.random.random(3))
print(np.random.random(3))
print(np.random.random(3))
print(np.random.random(3))
print(np.random.random(3))
print(np.random.random(3))

np.random.seed(1107)

print(np.random.random(3)+1)
print(np.random.random(3)+1)
print(np.random.random(3)+1)
print(np.random.random(3)+1)
print(np.random.random(3)+1)
print(np.random.random(3)+1)

