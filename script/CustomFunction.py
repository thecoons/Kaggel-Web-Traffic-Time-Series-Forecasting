import pandas as pd
import matplotlib.pyplot as plt
import re
import random as rdm


def load_data_ananas(file_path):
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
def sample_data_row_ananas(df, size_row, seed=rdm.random()):
    rdm.seed(seed)
    arr_random = [rdm.randint(0, df.shape[1]-(1 + size_row)) for i in range(df.shape[0])]
    df_sample = pd.DataFrame(columns=[i for i in range(size_row)])
    for i in range(df.shape[0]):
        sample = df.iloc[i, arr_random[i]: arr_random[i] + size_row]
        df_sample.loc[i] = sample.values

    return df_sample


def select_by_contains_page_ananas(df, pattern):
    return df.loc[lambda df: df.Page.str.contains(pattern), :]


def get_language_ananas(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[:2]

    return 'media'
