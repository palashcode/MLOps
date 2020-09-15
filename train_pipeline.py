import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

import config
import extract
import joblib
import time
import preprocessors as pp

def run_training():
    """Train the model."""
    # extract required data from 2 csv file and save
    data = extract.run_extract()

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.2,
        random_state=0)  # we are setting the seed here

    rf = RandomForestClassifier(n_jobs=2)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    if accuracy > 0.5:
        joblib.dump(rf, "models/"+time.strftime("%Y%m%d%H%M%S")+".joblib")
        print(f"Model accuracy:{accuracy} (Model Saved at ./models)")
    else:
        print(f"Model accuracy:{accuracy} (Model Not Saved)")


if __name__ == '__main__':
    run_training()
