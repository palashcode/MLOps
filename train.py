import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
import extract
import joblib
import json

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

    rf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    with open("models/version", "r") as fp:
        try:
            version = int(fp.read().strip())
        except:
            version = 0
    

    # matrics calculation
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    all_matrics = {"version":version+1, "accuracy":round(accuracy,4), "f1":round(f1,4)}#, "precision":precision, "recall":recall}
    with open(f"models/matrics/matrics_v{version+1}.json", "w") as fp:
        json.dump(all_matrics, fp)

    if(accuracy >=0.5 and f1 >= 0.5):
        joblib.dump(rf, f"models/model_v{version+1}")
        with open("models/version", "w") as fp:
            fp.write(str(version+1))
        print(f"New model saved models/model_v{version+1}")
        print(all_matrics)
    else:
        print("Model not better than previous so not updated.")


if __name__ == '__main__':
    run_training()
