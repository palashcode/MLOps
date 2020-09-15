import numpy as np
import pandas as pd

# categorical fill missing value 
def fill_missing_categorical(X, variables=None):
    if not isinstance(variables, list):
        variables = [variables]

    X = X.copy()
    for feature in variables:
        most_frequent = X[feature].mode().to_string().split()[1]
        X[feature].fillna(most_frequent, inplace=True)
    return X

# string to numbers categorical encoder
def encode_categorical(X,variables=None):
    if not isinstance(variables, list):
        variables = [variables]

    # encode feature as numeric values 
    from sklearn import preprocessing
    X = X.copy()
    for feature in variables:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(X[feature].apply(str))
        X[feature] = encoder.transform(X[feature].apply(str))

    return X

def normalize(X, variables=None):
    if not isinstance(variables, list):
        variables = [variables]

    X = X.copy()
    for feature in variables:
        # normalize features to 0 to 1
        X[feature] = (X[feature] - X[feature].min()) / (X[feature].max() - X[feature].min())

    return X

def drop_features(X, variables=None):
    if not isinstance(variables, list):
        variables = [variables]

    # encode labels
    X = X.copy()
    X = X.drop(variables, axis=1)

    return X