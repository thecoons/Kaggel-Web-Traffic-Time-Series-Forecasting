'''Fonctions pour web trafic wiki.'''
import math
import random as rdm
import re
# import logging

# import matplotlib.pyplot as plt
import pandas as pd
from numba import jit


def load_data_ananas(file_path):
    '''Charge les données issues du CSV.'''
    # Loadind csv train file and drop row where 'NaN' is in.
    df_train = pd.read_csv(file_path).dropna()

    # Itr on scoring columns
    for col in df_train.columns[1:]:
        # Cast float score to integer score, that safe 50% memory usage.
        df_train[col] = pd.to_numeric(df_train[col], downcast='integer')

    # Add column 'lang'
    df_train.insert(1, "lang", df_train.Page.map(get_language_ananas))

    return df_train


# Effectue un sample de taille de donnée et d'index aléatoire sur chaque row.
# Système de seed pour la génération afin d'évaluer
def sample_data_row_ananas(dframe, size_row, seed=rdm.random()):
    '''Créer une selection partielle des series temporelles.'''
    rdm.seed(seed)
    arr_random = [rdm.randint(0, dframe.shape[1]-(1 + size_row)) for i in range(dframe.shape[0])]
    df_sample = pd.DataFrame(columns=[i for i in range(size_row)])
    for i in range(dframe.shape[0]):
        sample = dframe.iloc[i, arr_random[i]: arr_random[i] + size_row]
        df_sample.loc[i] = sample.values

    return df_sample

def sample_data_row_banana(dframe_row, size_row):
    '''Créer une selection partielle des temportelles pour une unique time serie.'''
    df_sample = pd.DataFrame(columns=[i for i in range(size_row)])
    for i in range(dframe_row.shape[0] - (1 + size_row)):
        sample = dframe_row.iloc[i:i+size_row]
        df_sample.loc[i] = sample.values

    return df_sample


def select_by_contains_page_ananas(dframe, pattern):
    """Selectionne les row contenant le pattern donné."""
    return dframe.loc[lambda dframe: dframe.Page.str.contains(pattern), :]


def get_language_ananas(page):
    """Extrait la nationalitée de la page wiki."""
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[:2]

    return 'media'

@jit
def smap_fast(y_true, y_pred):
    """Function de scoring SMAP."""
    out = 0
    for i in range(y_true.shape[0]):
        val_true = y_true[i]
        val_pred = y_pred[i]
        val_sum = val_true + val_pred
        if val_sum == 0:
            continue
        out += math.fabs(val_true - val_pred) / val_sum
    out *= (200.0 / y_true.shape[0])

    return out
