import pandas as pd
import matplotlib.pyplot as plt
import re


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


def select_by_contains_page_ananas(df, pattern):
    return df.loc[lambda df: df.Page.str.contains(pattern), :]


def get_language_ananas(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        return res.group(0)[:2]
        
    return 'media'
