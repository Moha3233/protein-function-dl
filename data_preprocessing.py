import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


AMBIGUOUS = set('XBZJUO')


def clean_sequence(seq):
if len(seq) < 50:
return None
if any(aa in AMBIGUOUS for aa in seq):
return None
return seq


def load_and_clean(csv_path):
df = pd.read_csv(csv_path)
df['sequence'] = df['sequence'].apply(clean_sequence)
df = df.dropna()
return df


def split_data(X, y, seed=42):
X_train, X_temp, y_train, y_temp = train_test_split(
X, y, test_size=0.3, random_state=seed, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)
return X_train, X_val, X_test, y_train, y_val, y_test
