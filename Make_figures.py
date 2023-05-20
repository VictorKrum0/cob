import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

with open('ModelComparisonReport_save.txt', 'r') as file :
    lines = file.readlines()

KNN_scores = [float(lines[6+8*i].strip().split(' ')[-1]) for i in range(13)]
XGB_scores = [float(lines[7+8*i].strip().split(' ')[-1]) for i in range(13)]
MLP_scores = [float(lines[8+8*i].strip().split(' ')[-1]) for i in range(13)]

print(f'KNN untrained score : {sum(KNN_scores[1:13])/12}')
print(f'XGB : {sum(XGB_scores[1:13])/12}')
print(f'MLP : {sum(MLP_scores[1:13])/12}')

Mixed_scores = [float(lines[3+8*i].strip().split(':')[-1].strip('.')) for i in range(13)]

plt.scatter(range(13),KNN_scores, color='dimgrey',label='KNN')
plt.scatter(range(13),XGB_scores, color='darkgray',label='XGB')
plt.scatter(range(13),MLP_scores, color='lightgray',label='MLP')
plt.scatter(range(13),Mixed_scores, label='Mixed')

print(Mixed_scores)
plt.legend()
plt.show()
'''

with open('PCALog_50fold.txt', 'r') as file :
    lines = file.readlines()

KNN_scores = [float(lines[4+8*i].strip().split(' ')[-1]) for i in range(9)]
XGB_scores = [float(lines[6+8*i].strip().split(' ')[-1]) for i in range(9)]
MLP_scores = [float(lines[8+8*i].strip().split(' ')[-1]) for i in range(9)]

print(f'KNN untrained score : {sum(KNN_scores[1:4])/3}')
print(f'XGB : {sum(XGB_scores[1:4])/3}')
print(f'MLP : {sum(MLP_scores[1:4])/3}')

pca_sizes_r = [8,16,24,32,40,48,56,64,128]

pca_sizes = [8,12,13,14,15,16,17,18,19,20,21,22,23,24,32,40,48,56,64,128]

with open('PCALog_fine.txt', 'r') as file :
    lines = file.readlines()

KNN_scores = KNN_scores[0:1] + [float(lines[4+8*i].strip().split(' ')[-1]) for i in range(12)] + KNN_scores[2:]
XGB_scores = XGB_scores[0:1] + [float(lines[6+8*i].strip().split(' ')[-1]) for i in range(12)] + XGB_scores[2:]
MLP_scores = MLP_scores[0:1] + [float(lines[8+8*i].strip().split(' ')[-1]) for i in range(12)] + MLP_scores[2:]


#print(len(KNN_scores), len(pca_sizes))

plt.scatter(pca_sizes,KNN_scores, color='dimgrey',label='KNN')
plt.scatter(pca_sizes,XGB_scores, color='darkgray',label='XGB')
plt.scatter(pca_sizes,MLP_scores, color='lightgray',label='MLP')

plt.plot(pca_sizes,(np.array(MLP_scores)+np.array(KNN_scores)+np.array(XGB_scores))*0.33, color='steelblue',label='Average')

plt.xlabel('PCA Size')
plt.ylabel('Accuracy')
plt.xticks(pca_sizes_r)

with open('PCALog.txt', 'r') as file :
    lines = file.readlines()

Mixed_scores = [float(lines[4+4*i].strip().split(' ')[-1]) for i in range(12)]

#plt.scatter(range(12,24),Mixed_scores, c='steelblue',label='Mixed Model', marker='*')

plt.legend()
plt.show()


with open('PCALog_fine.txt', 'r') as file :
    lines = file.readlines()'''


