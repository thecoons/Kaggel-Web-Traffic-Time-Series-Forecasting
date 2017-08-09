'''Testing File.'''
# pylint: disable=C0103

# import pandas as pd
# import matplotlib.pyplot as plt
import logging
import script.CustomFunction as cf
from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.metrics import explained_variance_score

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s => %(message)s")

logging.info('Initialisation du script !!!')

df_train = cf.load_data_ananas('data/train_exp.csv')

logging.info('Data loaded !')
logging.debug('Data Head :\n %s', df_train.head())


df_train = cf.sample_data_row_banana(df_train.iloc[0, 2:], 5)

logging.info('Data sampled !')
logging.debug('Data_sampled :\n %s', df_train)

X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:, :-1], df_train.iloc[:, -1:],
                                                    test_size=0.2)

logging.info('Data splited !')
logging.debug('X_train :\n %s', X_train)
logging.debug('y_train :\n %s', y_train)
# logging.debug('y_train :\n %s', y_train.values)

# print(train.iloc[:, -1:].values.flatten())
# train.info()
# print(train.head())
# print(train.iloc[:10, :], train.iloc[:10, -1:])
# clf = svm.SVR()
# clf.fit(X_train.iloc[:, :-1].values, X_train.iloc[:, -1:].values.flatten())

# y_pred = clf.predict(y_test.iloc[:, :-1])

# report = explained_variance_score(y_test.iloc[:, -1:], y_pred)

# print('Score : {0}'.format(report))

# print(X_train.head())
