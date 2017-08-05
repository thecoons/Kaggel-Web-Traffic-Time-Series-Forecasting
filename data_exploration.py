'''Testing File.'''
# pylint: disable=C0103

# import pandas as pd
# import matplotlib.pyplot as plt
import script.CustomFunction as cf
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import explained_variance_score

df_train = cf.load_data_ananas('data/train_1.csv')

df_train = cf.sample_data_row_ananas(df_train.iloc[:, 2:], 5, 10)

X_train, X_test, y_train, y_test = train_test_split(df_train, test_size=0.2)

# print(train.iloc[:, -1:].values.flatten())
# train.info()
# print(train.head())
# print(train.iloc[:10, :], train.iloc[:10, -1:])
clf = svm.SVR()
clf.fit(X_train.iloc[:, :-1].values, X_train.iloc[:, -1:].values.flatten())

y_pred = clf.predict(y_test.iloc[:, :-1])

report = explained_variance_score(y_test.iloc[:, -1:], y_pred)

print('Score : {0}'.format(report))
