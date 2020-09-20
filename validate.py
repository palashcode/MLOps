import json
import sys
import pandas as pd
import config
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def validate_model():
    #load latest model
    with open("models/version", "r") as fp:
        version = fp.read()
    model = joblib.load(f"models/model_v{version}")
    # load data 
    data = pd.read_csv(config.VALIDATION_DATA_FILE)
    x = data[config.FEATURES]
    y = data[config.TARGET]
    y_pred = model.predict(x)
    # calculate matrics
    accuracy = accuracy_score(y,y_pred)
    f1 = f1_score(y,y_pred)
    if accuracy >=0.5 and f1 >= 0.5:
        return False
    return True

if __name__ == "__main__":
    validate_model()