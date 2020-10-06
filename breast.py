# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

type(cancer)

cancer.keys()

cancer['data']

type(cancer['data'])

cancer['target']

cancer['target_names']
print(cancer['DESCR'])
print(cancer['feature_names'])
print(cancer['filename'])
df = pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
df.head()
df.isnull().sum()#so we dont have nan values

# pair plot of sample feature
sns.pairplot(df, hue = 'target', 
             vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'] )

sns.countplot(df['target'])#using this graph we can see that 0 is malignant
plt.figure(figsize=(20,8))
sns.countplot(df['mean radius'])

X = df.drop(['target'],axis=1)
y = df['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

dt2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt2.predict(X_test_sc)
accuracy_score(y_test, y_pred_dt_sc)
