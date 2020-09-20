import pandas as pd
import numpy as np
import joblib
import config
import os

def make_prediction(data):
    #load latest model
    with open("models/version", "r") as fp:
        version = fp.read()
    model = joblib.load(f"models/model_v{version}")
    # predict
    data = data[config.FEATURES]
    results = model.predict(data)
    return results
   
if __name__ == '__main__':
    # test pipeline
    data = pd.read_csv(config.VALIDATION_DATA_FILE)
    pred = make_prediction(data)
    print(pred)
