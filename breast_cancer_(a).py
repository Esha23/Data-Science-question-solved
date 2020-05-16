# -*- coding: utf-8 -*-
"""
Created on Wed May  6 00:06:50 2020

@author: hp
"""

import numpy as np
import pandas as pd

data_read = pd.read_excel('BreastCancer_Prognostic_v1.xlsx')
data = np.array(data_read)
print (data_read.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

Y = data[:,1]
f1 = data[:, 0]
f2 = data[:, 2]
f3 = data[:, 3]
f4 = data[:, 4]
f5 = data[:, 5]
f6 = data[:, 6]
f7 = data[:, 7]
f8 = data[:, 8]
f9 = data[:, 9]
f10 = data[:, 10]
f11 = data[:, 11]
f12 = data[:, 12]
f13 = data[:, 13]
f14 = data[:, 14]
f15 = data[:, 15]
f16 = data[:, 16]
f17 = data[:, 17]
f18 = data[:, 18]
f19 = data[:, 19]
f20 = data[:, 20]
f21 = data[:, 21]
f22 = data[:, 22]
f23 = data[:, 23]
f24 = data[:, 24]
f25 = data[:, 25]
f26 = data[:, 26]
f27 = data[:, 27]
f28 = data[:, 28]
f29 = data[:, 29]
f30 = data[:, 30]
f31 = data[:, 31]
f32 = data[:, 32]
f33 = data[:, 33]
f34 = data[:, 34]

l = len(f34)
for i in range(l):
    if(f34[i] == '?'):
        f34[i]=0
    
X = np.array(list(zip(f1, f3, f4, f7, f8, f9, f11, f12, f13, f14, f17, f18, f19, f20, f21, f22, f24, f26, f27, f28, f29, f30, f31, f33, f34)), dtype = np.float32)


df = pd.DataFrame(data)
print(df)

print(df[1].value_counts())
df_majority = df[df[1] == 'N']
df_minority = df[df[1] == 'R']

df_minority_upsampled = resample(df_minority, 
                                  replace=True,
                                  n_samples=151,
                                  random_state=4)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled[1].value_counts())

y = df_upsampled[1]
X = df_upsampled.drop(1, axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30)
train_test_split(Y, shuffle=False)

df = pd.DataFrame(Y)
print(df)
print(df[0].value_counts())
y = Y_train
X = X_train

clf_4 = RandomForestClassifier()
clf_4.fit(X, y)

pred_y_4 = clf_4.predict(X_test)
print(Y_test)
print(pred_y_4)
 
print(np.unique(pred_y_4))
 
print(accuracy_score(Y_test, pred_y_4))

prob_y_4 = clf_4.predict_proba(X_test)
prob_y_4 = [p[1] for p in prob_y_4]
print(roc_auc_score(Y_test, prob_y_4))


print("Confusion Matrix: \n",confusion_matrix(Y_test, pred_y_4),"\n")
print("Classification Accuracy: \n",classification_report(Y_test, pred_y_4),"\n")

df = pd.DataFrame(data)
print(df)
df[2] =df[2].astype('category').cat.codes
df[3] =df[3].astype('category').cat.codes
df[4] =df[4].astype('category').cat.codes
# df[5] =df[5].astype('category').cat.codes
# df[6] =df[6].astype('category').cat.codes
df[7] =df[7].astype('category').cat.codes
df[8] =df[8].astype('category').cat.codes
df[9] =df[9].astype('category').cat.codes
# df[10] =df[10].astype('category').cat.codes
df[11] =df[11].astype('category').cat.codes
df[12] =df[12].astype('category').cat.codes
df[13] =df[13].astype('category').cat.codes
df[14] =df[14].astype('category').cat.codes
# df[15] =df[15].astype('category').cat.codes
# df[16] =df[16].astype('category').cat.codes
df[17] =df[17].astype('category').cat.codes
df[18] =df[18].astype('category').cat.codes
df[19] =df[19].astype('category').cat.codes
df[20] =df[20].astype('category').cat.codes
df[21] =df[21].astype('category').cat.codes
df[22] =df[22].astype('category').cat.codes
# df[23] =df[23].astype('category').cat.codes
df[24] =df[24].astype('category').cat.codes
# df[25] =df[25].astype('category').cat.codes
# df[26] =df[26].astype('category').cat.codes
df[27] =df[27].astype('category').cat.codes
df[28] =df[28].astype('category').cat.codes
df[29] =df[29].astype('category').cat.codes
df[30] =df[30].astype('category').cat.codes
df[31] =df[31].astype('category').cat.codes
# df[32] =df[32].astype('category').cat.codes
df[33] =df[33].astype('category').cat.codes
df[34] =df[34].astype('category').cat.codes

corr_matrix = df.corr()
fig = plt.figure(figsize=(23, 23))
corr_matrix[2].sort_values(ascending=False)
plt.plot((corr_matrix))
plt.show()

import seaborn as sns
plt.figure(figsize=(12,10))

fig = plt.figure(figsize=(20, 20))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

df = pd.DataFrame(X_test)
df['target_values'] = Y_test
df['predicted_values'] = pred_y_4
print(df)
filepath = 'prediction2.xlsx'
df.to_excel(filepath, index=False)
