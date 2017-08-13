'''Testing File.'''
# pylint: disable=C0103

# import pandas as pd
# import matplotlib.pyplot as plt
import logging
import math

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import script.CustomFunction as cf

# from sklearn import svm
# from sklearn.metrics import explained_variance_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s => %(message)s")

logging.info('Initialisation du script !!!')

LOOK_BACK = 4

df_train_brut = cf.load_data_ananas('data/train_exp.csv')
logging.info('Data loaded !')
logging.debug('Data Head :\n %s', df_train_brut.head())


df_train = cf.sample_data_row_banana(df_train_brut.iloc[0, 2:], LOOK_BACK + 1)
logging.info('Data sampled !')
logging.debug('Data_sampled :\n %s', df_train)

X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:, :-1], df_train.iloc[:, -1:],
                                                    test_size=0.2)
logging.info('Data splited !')
logging.debug('X_train :\n %s', X_train)
logging.debug('y_train :\n %s', y_train)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.values)
y_train = scaler.fit_transform(y_train.values)

X_test = scaler.fit_transform(X_test.values)
y_test = scaler.fit_transform(y_test.values)

logging.info('Data normalize !')
logging.debug('Data_norm X_train :\n %s', X_train)
logging.debug('Data_norm y_train :\n %s', y_train)

model = cf.model_lstm_ananas(LOOK_BACK)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

train_score = math.sqrt(mean_squared_error(y_train, train_predict[:, 0]))
test_score = math.sqrt(mean_squared_error(y_test, test_predict[:, 0]))

logging.info('Train Score %s RMSE', train_score)
logging.info('Test Score %s RMSE', test_score)

# df_train_brut.rename(columns=[i for i in range(df_train_brut.shape[1])])
df_train_brut.columns = [i for i in range(df_train_brut.shape[1])]
cf.draw_plot(df_train_brut.iloc[2, 2:])


# logging.debug('Unormalize :\n %s', cf.data_unscalling_ananas(y_train, scaler))

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
