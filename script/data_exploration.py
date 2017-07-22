import pandas as pd
import matplotlib.pyplot as plt
import script.CustomFunction as cf
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_absolute_error

df_train = cf.load_data_ananas('data/train_1.csv')

train, test = train_test_split(df_train.iloc[:, 2:], test_size=0.2)

train.info()
# print(train.head())

clf = svm.SVC()
clf.fit(train.iloc[:, :-1], train.iloc[:, -1:])

y_pred = clf.predict(test.iloc[:, :-1])

report = mean_absolute_error(test.iloc[:, -1:])

print('Score : {0}'.format(report))
